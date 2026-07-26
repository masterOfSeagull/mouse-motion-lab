from __future__ import annotations

import json
from math import atan2, degrees, hypot, log2
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from mouselearn.domain.collection import (
    CaptureHealthRecord,
    CollectionPhaseMarker,
    CollectionSessionPlan,
    RawEventFileReference,
    TrialFinalization,
    TrialPlan,
)
from mouselearn.domain.dataset import DatasetSnapshotPlan, session_held_out_assignments


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Repositories:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def set_status(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO application_status(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at", (key, value, utcnow()))

    def get_status(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM application_status WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_setting(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT INTO settings(key,value_json,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at", (key, json.dumps(value), utcnow()))

    def get_setting(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def audit(self, entity_type: str, entity_id: str, action: str, detail: dict[str, Any] | None = None) -> str:
        event_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO audit_events(id,entity_type,entity_id,action,detail_json,created_at) VALUES(?,?,?,?,?,?)", (event_id, entity_type, entity_id, action, json.dumps(detail or {}), utcnow()))
        return event_id

    def create_job(self, job_type: str) -> str:
        job_id, now = str(uuid.uuid4()), utcnow()
        self.conn.execute("INSERT INTO jobs(id,type,status,created_at,updated_at) VALUES(?,?,?,?,?)", (job_id, job_type, "queued", now, now))
        self.audit("job", job_id, "queued", {"type": job_type})
        return job_id

    def update_job(self, job_id: str, status: str | None = None, progress: int | None = None, stage: str | None = None, error: str | None = None) -> None:
        row = self.conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        now = utcnow()
        fields, values = ["updated_at=?"], [now]
        if status is not None:
            fields.append("status=?"); values.append(status)
            if status == "running": fields.append("started_at=COALESCE(started_at,?)"); values.append(now)
            if status in {"completed", "failed", "cancelled"}: fields.append("finished_at=?"); values.append(now)
        if progress is not None: fields.append("progress=?"); values.append(progress)
        if stage is not None: fields.append("stage=?"); values.append(stage)
        if error is not None: fields.append("error=?"); values.append(error)
        values.append(job_id)
        self.conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", values)
        if status is not None:
            self.audit("job", job_id, status, {"progress": progress, "stage": stage, "error": error})

    def job(self, job_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None: raise KeyError(job_id)
        return dict(row)

    def job_totals(self) -> dict[str, int]:
        counts = {state: 0 for state in ("queued", "running", "completed", "failed", "cancelled")}
        for row in self.conn.execute("SELECT status, count(*) AS total FROM jobs GROUP BY status"):
            counts[row[0]] = row[1]
        return counts

    def reconcile_interrupted_jobs(self) -> int:
        rows = self.conn.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
        for row in rows:
            self.update_job(row[0], status="failed", error="interrupted before application startup")
        return len(rows)

    def create_collection_session(self, plan: CollectionSessionPlan) -> str:
        """Create planned session metadata before collection can begin."""
        session_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                "INSERT INTO recording_sessions(id,status,created_at) VALUES(?,?,?)",
                (session_id, "planned", now),
            )
            self.conn.execute(
                """INSERT INTO collection_session_details(
                    session_id,display_name,mode,state,planned_trials,random_seed,config_json,environment_json,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, plan.display_name, plan.mode, "planned", plan.planned_trials, plan.random_seed,
                    json.dumps(plan.config, sort_keys=True), json.dumps(plan.environment, sort_keys=True), now,
                ),
            )
            self.audit("collection_session", session_id, "planned", {"mode": plan.mode, "planned_trials": plan.planned_trials})
        return session_id

    def transition_collection_session(self, session_id: str, state: str, failure_reason: str | None = None) -> None:
        """Advance a session through its explicit lifecycle and mirror legacy status."""
        allowed = {
            "planned": {"active", "failed", "cancelled"},
            "active": {"paused", "completed", "failed", "cancelled"},
            "paused": {"active", "failed", "cancelled"},
            "completed": set(), "failed": set(), "cancelled": set(),
        }
        row = self.conn.execute("SELECT state FROM collection_session_details WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown collection session: {session_id}")
        current = row[0]
        if state not in allowed[current]:
            raise ValueError(f"invalid collection session transition: {current} -> {state}")
        now = utcnow()
        terminal = state in {"completed", "failed", "cancelled"}
        legacy_state = {"planned": "planned", "active": "active", "paused": "active", "completed": "completed", "failed": "discarded", "cancelled": "discarded"}[state]
        with self.conn:
            self.conn.execute(
                """UPDATE collection_session_details
                   SET state=?, failure_reason=?, started_at=CASE WHEN ?='active' THEN COALESCE(started_at, ?) ELSE started_at END,
                       ended_at=CASE WHEN ? THEN ? ELSE ended_at END, updated_at=? WHERE session_id=?""",
                (state, failure_reason, state, now, terminal, now, now, session_id),
            )
            self.conn.execute(
                "UPDATE recording_sessions SET status=?, finished_at=CASE WHEN ? THEN ? ELSE finished_at END WHERE id=?",
                (legacy_state, terminal, now, session_id),
            )
            self.audit("collection_session", session_id, state, {"failure_reason": failure_reason})

    def collection_session(self, session_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT s.id, s.status AS legacy_status, s.created_at, s.finished_at,
                      d.display_name, d.mode, d.state, d.planned_trials, d.random_seed,
                      d.config_json, d.environment_json, d.failure_reason, d.started_at, d.ended_at, d.updated_at
               FROM recording_sessions AS s
               JOIN collection_session_details AS d ON d.session_id=s.id WHERE s.id=?""",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        result["environment"] = json.loads(result.pop("environment_json"))
        return result

    def update_collection_environment(self, session_id: str, values: dict[str, Any]) -> None:
        session = self.collection_session(session_id)
        environment = {**session["environment"], **values}
        with self.conn:
            self.conn.execute(
                "UPDATE collection_session_details SET environment_json=?, updated_at=? WHERE session_id=?",
                (json.dumps(environment, sort_keys=True), utcnow(), session_id),
            )
            self.audit("collection_session", session_id, "environment_updated", {"keys": sorted(values)})

    def create_trial(self, plan: TrialPlan) -> str:
        session_id = str(plan.session_id)
        session = self.collection_session(session_id)
        if session["state"] != "active":
            raise RuntimeError("trials can only be created for an active collection session")
        trial_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                "INSERT INTO trials(id,session_id,status,created_at) VALUES(?,?,?,?)",
                (trial_id, session_id, "target_visible", now),
            )
            self.conn.execute(
                """INSERT INTO trial_details(trial_id,condition_json,state,target_appeared_ns,start_screen_x,start_screen_y)
                   VALUES(?,?,?,?,?,?)""",
                (
                    trial_id, json.dumps(plan.condition.model_dump(mode="json"), sort_keys=True), "target_visible",
                    plan.target_appeared_ns, plan.start_screen_x, plan.start_screen_y,
                ),
            )
            self.audit("trial", trial_id, "target_visible", {"session_id": session_id})
        return trial_id

    def finalize_trial(self, trial_id: str, finalization: TrialFinalization) -> None:
        row = self.conn.execute("SELECT target_appeared_ns FROM trial_details WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown trial: {trial_id}")
        target_appeared_ns = row[0]
        if finalization.ended_ns < target_appeared_ns or any(click.timestamp_ns < target_appeared_ns for click in finalization.clicks):
            raise ValueError("trial timestamps cannot precede target appearance")
        first_click_ns = next((click.timestamp_ns for click in finalization.clicks), None)
        valid_click_ns = next((click.timestamp_ns for click in finalization.clicks if click.is_valid), None)
        with self.conn:
            self.conn.execute(
                """UPDATE trial_details SET state=?, first_click_ns=?, valid_click_ns=?, ended_ns=?, end_reason=?, clicks_json=?
                   WHERE trial_id=?""",
                (
                    finalization.state, first_click_ns, valid_click_ns, finalization.ended_ns, finalization.end_reason,
                    json.dumps([click.model_dump(mode="json") for click in finalization.clicks]), trial_id,
                ),
            )
            self.conn.execute("UPDATE trials SET status=? WHERE id=?", (finalization.state, trial_id))
            self.audit("trial", trial_id, finalization.state, {"end_reason": finalization.end_reason, "click_count": len(finalization.clicks)})

    def trial(self, trial_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT t.id, t.session_id, t.status, t.created_at, d.condition_json, d.state,
                      d.target_appeared_ns, d.start_screen_x, d.start_screen_y,
                      d.first_click_ns, d.valid_click_ns, d.ended_ns, d.end_reason, d.clicks_json
               FROM trials AS t JOIN trial_details AS d ON d.trial_id=t.id WHERE t.id=?""",
            (trial_id,),
        ).fetchone()
        if row is None:
            raise KeyError(trial_id)
        result = dict(row)
        result["condition"] = json.loads(result.pop("condition_json"))
        result["clicks"] = json.loads(result.pop("clicks_json"))
        return result

    def record_raw_event_file(self, session_id: str, reference: RawEventFileReference) -> str:
        self.collection_session(session_id)
        file_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                """INSERT INTO raw_event_files(
                    id,session_id,relative_path,format,status,event_count,first_timestamp_ns,last_timestamp_ns,
                    qpc_frequency_hz,byte_count,sha256,created_at,finalized_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    file_id, session_id, reference.relative_path, "parquet", "complete", reference.event_count,
                    reference.first_timestamp_ns, reference.last_timestamp_ns, reference.qpc_frequency_hz,
                    reference.byte_count, reference.sha256, now, now,
                ),
            )
            self.audit("raw_event_file", file_id, "recorded", {"session_id": session_id, "event_count": reference.event_count})
        return file_id

    def record_capture_health(self, session_id: str, record: CaptureHealthRecord) -> str:
        self.collection_session(session_id)
        event_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                "INSERT INTO capture_health_events(id,session_id,severity,code,occurred_at_ns,detail_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (event_id, session_id, record.severity, record.code, record.occurred_at_ns, json.dumps(record.detail, sort_keys=True), now),
            )
            self.audit("collection_session", session_id, f"capture_{record.severity}", {"code": record.code})
        return event_id

    def record_phase_marker(self, session_id: str, marker: CollectionPhaseMarker, trial_id: str | None = None) -> str:
        self.collection_session(session_id)
        if trial_id is not None:
            trial = self.conn.execute("SELECT session_id FROM trials WHERE id=?", (trial_id,)).fetchone()
            if trial is None:
                raise KeyError(f"unknown trial: {trial_id}")
            if trial[0] != session_id:
                raise ValueError("phase marker trial does not belong to session")
        marker_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                """INSERT INTO collection_phase_markers(
                    id,session_id,trial_id,phase,timestamp_ns,screen_x,screen_y,detail_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    marker_id, session_id, trial_id, marker.phase, marker.timestamp_ns, marker.screen_x, marker.screen_y,
                    json.dumps(marker.detail, sort_keys=True), now,
                ),
            )
            self.audit("collection_phase_marker", marker_id, marker.phase, {"session_id": session_id, "trial_id": trial_id})
        return marker_id

    def phase_markers_for_trial(self, trial_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM collection_phase_markers WHERE trial_id=? ORDER BY timestamp_ns", (trial_id,)
        )]

    def set_collection_quality(self, session_id: str, classification: str, reasons: list[str]) -> None:
        if classification not in {"legacy", "current"}:
            raise ValueError("collection quality must be legacy or current")
        self.collection_session(session_id)
        with self.conn:
            self.conn.execute(
                """INSERT INTO collection_session_quality(session_id,classification,reasons_json,assessed_at) VALUES(?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET classification=excluded.classification,
                   reasons_json=excluded.reasons_json, assessed_at=excluded.assessed_at""",
                (session_id, classification, json.dumps(reasons, sort_keys=True), utcnow()),
            )
            self.audit("collection_session", session_id, f"quality_{classification}", {"reasons": reasons})

    def reconcile_legacy_collection_data(self) -> int:
        """Preserve older recordings while replacing centre-based geometry with recoverable realized values."""
        rows = self.conn.execute(
            """SELECT t.id, t.session_id, d.condition_json, d.start_screen_x, d.start_screen_y
               FROM trials AS t JOIN trial_details AS d ON d.trial_id=t.id
               JOIN collection_session_quality AS q ON q.session_id=t.session_id
               WHERE q.classification='legacy'"""
        ).fetchall()
        updated_sessions: set[str] = set()
        with self.conn:
            for row in rows:
                condition = json.loads(row["condition_json"])
                if condition.get("collection_protocol_version", 1) >= 2:
                    continue
                start_x, start_y = row["start_screen_x"], row["start_screen_y"]
                if start_x is None or start_y is None:
                    continue
                dx, dy = condition["target_x"] - start_x, condition["target_y"] - start_y
                distance = hypot(dx, dy)
                condition.setdefault("requested_distance_px", condition.get("distance_px"))
                condition.setdefault("requested_radius_px", condition.get("radius_px"))
                condition.setdefault("requested_angle_degrees", condition.get("angle_degrees"))
                condition.setdefault("requested_screen_region", condition.get("screen_region"))
                condition["distance_px"] = distance
                condition["angle_degrees"] = degrees(atan2(dy, dx)) % 360 if distance else 0.0
                difficulty = log2(distance / (2 * condition["radius_px"]) + 1) if distance else 0.0
                condition["difficulty_band"] = "low" if difficulty < 2 else "medium" if difficulty < 4 else "high"
                condition["collection_protocol_version"] = 1
                condition["reaction_time_confidence"] = "legacy_render_unconfirmed"
                self.conn.execute("UPDATE trial_details SET condition_json=? WHERE trial_id=?", (json.dumps(condition, sort_keys=True), row["id"]))
                updated_sessions.add(row["session_id"])
            for session_id in updated_sessions:
                self.audit("collection_session", session_id, "legacy_realized_geometry_recomputed", {})
        return len(updated_sessions)

    def raw_event_files(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM raw_event_files WHERE session_id=? ORDER BY created_at", (session_id,))]

    def set_session_review(self, session_id: str, disposition: str, reason: str = "") -> None:
        if disposition not in {"retained", "discarded"}:
            raise ValueError("session review disposition must be retained or discarded")
        if len(reason) > 500:
            raise ValueError("review reason must be at most 500 characters")
        self.collection_session(session_id)
        with self.conn:
            self.conn.execute(
                """INSERT INTO collection_session_reviews(session_id,disposition,reason,reviewed_at) VALUES(?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET disposition=excluded.disposition, reason=excluded.reason, reviewed_at=excluded.reviewed_at""",
                (session_id, disposition, reason, utcnow()),
            )
            self.audit("collection_session", session_id, f"review_{disposition}", {"reason": reason})

    def set_trial_review(self, trial_id: str, disposition: str, reason: str = "") -> None:
        if disposition not in {"retained", "discarded"}:
            raise ValueError("trial review disposition must be retained or discarded")
        if len(reason) > 500:
            raise ValueError("review reason must be at most 500 characters")
        if self.conn.execute("SELECT 1 FROM trials WHERE id=?", (trial_id,)).fetchone() is None:
            raise KeyError(f"unknown trial: {trial_id}")
        with self.conn:
            self.conn.execute(
                """INSERT INTO trial_reviews(trial_id,disposition,reason,reviewed_at) VALUES(?,?,?,?)
                   ON CONFLICT(trial_id) DO UPDATE SET disposition=excluded.disposition, reason=excluded.reason, reviewed_at=excluded.reviewed_at""",
                (trial_id, disposition, reason, utcnow()),
            )
            self.audit("trial", trial_id, f"review_{disposition}", {"reason": reason})

    def collection_sessions_for_review(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT s.id, s.created_at, s.finished_at, d.display_name, d.state, d.planned_trials,
                      COALESCE(sr.disposition, 'retained') AS review_disposition,
                      COALESCE(sr.reason, '') AS review_reason,
                      count(t.id) AS trial_count,
                      sum(CASE WHEN t.status='completed' THEN 1 ELSE 0 END) AS completed_trials,
                      sum(CASE WHEN tr.disposition='discarded' THEN 1 ELSE 0 END) AS discarded_trials
               FROM recording_sessions AS s
               JOIN collection_session_details AS d ON d.session_id=s.id
               LEFT JOIN collection_session_reviews AS sr ON sr.session_id=s.id
               LEFT JOIN trials AS t ON t.session_id=s.id
               LEFT JOIN trial_reviews AS tr ON tr.trial_id=t.id
               GROUP BY s.id ORDER BY s.created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def trials_for_review(self, session_id: str) -> list[dict[str, Any]]:
        self.collection_session(session_id)
        rows = self.conn.execute(
            """SELECT t.id, t.status, t.created_at, d.condition_json, d.target_appeared_ns, d.start_screen_x, d.start_screen_y, d.first_click_ns,
                      d.valid_click_ns, d.ended_ns, d.end_reason, d.clicks_json,
                      COALESCE(tr.disposition, 'retained') AS review_disposition,
                      COALESCE(tr.reason, '') AS review_reason
               FROM trials AS t
               JOIN trial_details AS d ON d.trial_id=t.id
               LEFT JOIN trial_reviews AS tr ON tr.trial_id=t.id
               WHERE t.session_id=? ORDER BY d.target_appeared_ns""",
            (session_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            result = dict(row)
            result["condition"] = json.loads(result.pop("condition_json"))
            result["clicks"] = json.loads(result.pop("clicks_json"))
            results.append(result)
        return results

    def eligible_dataset_sessions(self) -> list[dict[str, Any]]:
        """Completed, retained sessions with finalized raw evidence and valid trials."""
        rows = self.conn.execute(
            """SELECT s.id, s.created_at, d.display_name, count(t.id) AS trial_count
               FROM recording_sessions AS s
               JOIN collection_session_details AS d ON d.session_id=s.id
               LEFT JOIN collection_session_reviews AS sr ON sr.session_id=s.id
               JOIN trials AS t ON t.session_id=s.id
               JOIN trial_details AS td ON td.trial_id=t.id
               LEFT JOIN trial_reviews AS tr ON tr.trial_id=t.id
               WHERE s.status='completed' AND d.state='completed'
                 AND COALESCE(sr.disposition, 'retained')='retained'
                 AND t.status='completed' AND td.valid_click_ns IS NOT NULL
                 AND COALESCE(tr.disposition, 'retained')='retained'
                 AND EXISTS(SELECT 1 FROM raw_event_files AS rf
                            WHERE rf.session_id=s.id AND rf.status='complete' AND rf.sha256 IS NOT NULL)
                 AND NOT EXISTS(SELECT 1 FROM raw_event_files AS rf
                                WHERE rf.session_id=s.id AND (rf.status <> 'complete' OR rf.sha256 IS NULL))
               GROUP BY s.id ORDER BY s.created_at, s.id"""
        ).fetchall()
        return [dict(row) for row in rows]

    def create_dataset_snapshot_draft(self, plan: DatasetSnapshotPlan, code_revision: str) -> dict[str, Any]:
        """Persist all immutable source choices while the manifest is being written."""
        eligible = self.eligible_dataset_sessions()
        eligible_by_id = {row["id"]: row for row in eligible}
        selected_ids = [str(value) for value in plan.session_ids] or [row["id"] for row in eligible]
        unavailable = [session_id for session_id in selected_ids if session_id not in eligible_by_id]
        if unavailable:
            raise ValueError("selected sessions are not eligible for a dataset snapshot: " + ", ".join(unavailable))
        assignments, warnings = session_held_out_assignments(selected_ids, plan.split)
        placeholders = ",".join("?" for _ in selected_ids)
        legacy_sessions = [row[0] for row in self.conn.execute(
            f"""SELECT s.id FROM recording_sessions AS s
                LEFT JOIN collection_session_quality AS q ON q.session_id=s.id
                WHERE s.id IN ({placeholders}) AND COALESCE(q.classification, 'legacy')='legacy'""",
            selected_ids,
        )]
        if legacy_sessions:
            warnings.append(
                "Legacy collection protocol selected: realized geometry was recomputed where possible; "
                "reaction-time values are not high-confidence ground truth."
            )
        trial_rows = self.conn.execute(
            f"""SELECT t.id, t.session_id
                FROM trials AS t
                JOIN recording_sessions AS s ON s.id=t.session_id
                JOIN trial_details AS td ON td.trial_id=t.id
                LEFT JOIN trial_reviews AS tr ON tr.trial_id=t.id
                WHERE t.session_id IN ({placeholders}) AND s.status='completed'
                  AND t.status='completed' AND td.valid_click_ns IS NOT NULL
                  AND COALESCE(tr.disposition, 'retained')='retained'
                ORDER BY s.created_at, td.target_appeared_ns, t.id""",
            selected_ids,
        ).fetchall()
        if not trial_rows:
            raise ValueError("selected sessions contain no retained valid-click trials")
        raw_rows = self.conn.execute(
            f"""SELECT session_id, relative_path, sha256, byte_count
                FROM raw_event_files
                WHERE session_id IN ({placeholders}) AND status='complete' AND sha256 IS NOT NULL
                ORDER BY session_id, relative_path""",
            selected_ids,
        ).fetchall()
        hashes_by_session: dict[str, list[str]] = {session_id: [] for session_id in selected_ids}
        raw_files: list[dict[str, Any]] = []
        for row in raw_rows:
            item = dict(row)
            hashes_by_session[item["session_id"]].append(item["sha256"])
            raw_files.append(item)
        missing_hashes = [session_id for session_id, hashes in hashes_by_session.items() if not hashes]
        if missing_hashes:
            raise ValueError("selected sessions have no finalized raw event hashes: " + ", ".join(missing_hashes))
        raw_session_hashes = [{"session_id": session_id, "sha256": hashes_by_session[session_id]} for session_id in selected_ids]
        ordered_trial_ids = [row["id"] for row in trial_rows]
        split_config = {"strategy": "session_held_out", **plan.split.model_dump(mode="json")}
        manifest = {
            "schema_version": 1,
            "name": plan.name,
            "description": plan.description,
            "ordered_trial_ids": ordered_trial_ids,
            "raw_session_hashes": raw_session_hashes,
            "raw_files": raw_files,
            "preprocessing_config": plan.preprocessing_config,
            "split_config": split_config,
            "split_assignments": assignments,
            "feature_schema_version": plan.feature_schema_version,
            "code_revision": code_revision,
            "warnings": warnings,
            "data_quality": {"legacy_session_ids": legacy_sessions, "reaction_time_high_confidence": not legacy_sessions},
        }
        snapshot_id, now = str(uuid.uuid4()), utcnow()
        with self.conn:
            self.conn.execute(
                "INSERT INTO dataset_snapshots(id,status,created_at) VALUES(?,?,?)", (snapshot_id, "draft", now),
            )
            self.conn.execute(
                """INSERT INTO dataset_snapshot_details(
                    snapshot_id,name,description,ordered_trial_ids_json,raw_session_hashes_json,
                    preprocessing_config_json,split_config_json,warnings_json,feature_schema_version,
                    code_revision,trial_count,session_count
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot_id, plan.name, plan.description, json.dumps(ordered_trial_ids), json.dumps(raw_session_hashes),
                    json.dumps(plan.preprocessing_config, sort_keys=True), json.dumps(split_config, sort_keys=True),
                    json.dumps(warnings), plan.feature_schema_version, code_revision, len(ordered_trial_ids), len(selected_ids),
                ),
            )
            self.conn.executemany(
                "INSERT INTO dataset_snapshot_trials(snapshot_id,trial_id,session_id,split,ordinal) VALUES(?,?,?,?,?)",
                [(snapshot_id, row["id"], row["session_id"], assignments[row["session_id"]], index) for index, row in enumerate(trial_rows)],
            )
            self.audit("dataset_snapshot", snapshot_id, "draft_created", {"trial_count": len(ordered_trial_ids), "session_count": len(selected_ids)})
        return {"id": snapshot_id, "manifest": manifest, "raw_files": raw_files}

    def finalize_dataset_snapshot(self, snapshot_id: str, manifest_relative_path: str, manifest_sha256: str) -> None:
        if not manifest_relative_path or manifest_relative_path.startswith("/") or ".." in manifest_relative_path.replace("\\", "/").split("/"):
            raise ValueError("manifest path must be relative to the data root")
        if len(manifest_sha256) != 64:
            raise ValueError("manifest hash must be a SHA-256 digest")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            updated = self.conn.execute(
                """UPDATE dataset_snapshot_details SET manifest_relative_path=?, manifest_sha256=?
                   WHERE snapshot_id=?""",
                (manifest_relative_path.replace("\\", "/"), manifest_sha256, snapshot_id),
            ).rowcount
            if updated != 1:
                raise KeyError(snapshot_id)
            status_updated = self.conn.execute(
                "UPDATE dataset_snapshots SET status='ready' WHERE id=? AND status='draft'", (snapshot_id,),
            ).rowcount
            if status_updated != 1:
                raise RuntimeError("dataset snapshot was not in draft state")
            self.audit("dataset_snapshot", snapshot_id, "ready", {"manifest_sha256": manifest_sha256})
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def discard_dataset_snapshot_draft(self, snapshot_id: str, reason: str) -> None:
        """Remove a never-ready draft after a manifest build failure; ready snapshots are immutable."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT status FROM dataset_snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if row is None:
                self.conn.execute("ROLLBACK")
                return
            if row[0] != "draft":
                raise RuntimeError("only draft snapshots can be discarded")
            self.audit("dataset_snapshot", snapshot_id, "draft_discarded", {"reason": reason})
            self.conn.execute("DELETE FROM dataset_snapshot_trials WHERE snapshot_id=?", (snapshot_id,))
            self.conn.execute("DELETE FROM dataset_snapshot_details WHERE snapshot_id=?", (snapshot_id,))
            self.conn.execute("DELETE FROM dataset_snapshots WHERE id=?", (snapshot_id,))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def dataset_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT s.id, s.status, s.created_at, d.name, d.description, d.ordered_trial_ids_json,
                      d.raw_session_hashes_json, d.preprocessing_config_json, d.split_config_json, d.warnings_json,
                      d.feature_schema_version, d.code_revision, d.manifest_sha256, d.manifest_relative_path,
                      d.trial_count, d.session_count
               FROM dataset_snapshots AS s JOIN dataset_snapshot_details AS d ON d.snapshot_id=s.id
               WHERE s.id=?""",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        result = dict(row)
        for name in ("ordered_trial_ids", "raw_session_hashes", "preprocessing_config", "split_config", "warnings"):
            result[name] = json.loads(result.pop(f"{name}_json"))
        result["splits"] = [dict(item) for item in self.conn.execute(
            "SELECT trial_id,session_id,split,ordinal FROM dataset_snapshot_trials WHERE snapshot_id=? ORDER BY ordinal", (snapshot_id,)
        )]
        return result

    def dataset_snapshots(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT s.id, s.status, s.created_at, d.name, d.trial_count, d.session_count,
                      d.manifest_sha256, d.manifest_relative_path, d.warnings_json
               FROM dataset_snapshots AS s JOIN dataset_snapshot_details AS d ON d.snapshot_id=s.id
               ORDER BY s.created_at DESC"""
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["warnings"] = json.loads(item.pop("warnings_json"))
            results.append(item)
        return results
