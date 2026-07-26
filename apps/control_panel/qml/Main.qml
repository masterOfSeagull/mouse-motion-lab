import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Window

ApplicationWindow {
    id: window
    width: 980
    height: 620
    visible: true
    title: "MouseMotionLab"
    color: "#f3f4f6"
    font.family: "Segoe UI"
    font.pixelSize: 14

    property int currentPage: 0
    RowLayout {
        anchors.fill: parent
        spacing: 0
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: 210
            color: "#1f2937"
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 18
                Label { text: "MouseMotionLab"; font.bold: true; font.pixelSize: 19; color: "white" }
                Label { text: "Milestone 2 in progress"; color: "#9ca3af" }
                Button { text: "Dashboard"; onClicked: currentPage = 0; Layout.fillWidth: true }
                Button { text: "Diagnostics"; onClicked: currentPage = 1; Layout.fillWidth: true }
                Button { text: "Jobs"; onClicked: currentPage = 2; Layout.fillWidth: true }
                Button { text: "Collection"; onClicked: currentPage = 3; Layout.fillWidth: true }
                Button { text: "Sessions"; onClicked: currentPage = 4; Layout.fillWidth: true }
                Item { Layout.fillHeight: true }
                Label { text: "Playback remains disabled"; wrapMode: Text.Wrap; color: "#9ca3af"; Layout.fillWidth: true }
            }
        }
        StackLayout {
            Layout.fillWidth: true; Layout.fillHeight: true; currentIndex: currentPage
            Item {
                Column { anchors.fill: parent; anchors.margins: 34; spacing: 14
                    Label { text: "Dashboard"; font.pixelSize: 28; font.bold: true }
                    Label { text: "Data root: " + appController.dataRoot; wrapMode: Text.Wrap; Layout.fillWidth: true }
                    Label { text: "SQLite schema: " + appController.schemaVersion }
                    Label { text: "Jobs — " + appController.jobSummary; Layout.fillWidth: true }
                    Button { text: "Refresh"; onClicked: appController.refresh() }
                }
            }
            Item {
                Column { anchors.fill: parent; anchors.margins: 34; spacing: 14
                    Label { text: "Diagnostics"; font.pixelSize: 28; font.bold: true }
                    Label { text: "Runs a subprocess check of storage, SQLite, Python, optional ML dependencies, and display metadata."; wrapMode: Text.Wrap; Layout.fillWidth: true }
                    Label { text: "Raw Input ready: " + (appController.collectionController.available ? "yes" : "no") }
                    Label { text: "Capture state: " + appController.collectionController.state + " · observed events: " + appController.collectionController.observedEvents }
                    Label { text: "Native buffer: " + appController.collectionController.bufferedEvents + " pending · overflows: " + appController.collectionController.overflowEvents + " · QPC: " + appController.collectionController.qpcFrequencyHz + " Hz" }
                    Button { text: "Run diagnostic"; onClicked: appController.startDiagnostic() }
                    Button { text: "Cancel current job"; onClicked: appController.cancelJob() }
                    Label { text: appController.jobMessage; wrapMode: Text.Wrap; Layout.fillWidth: true }
                }
            }
            Item {
                Column { anchors.fill: parent; anchors.margins: 34; spacing: 14
                    Label { text: "Jobs"; font.pixelSize: 28; font.bold: true }
                    Label { text: appController.jobSummary }
                    Label { text: "Latest worker status: " + appController.jobMessage; wrapMode: Text.Wrap; Layout.fillWidth: true }
                }
            }
            Item {
                id: collectionPage
                property var collector: appController.collectionController
                ColumnLayout { anchors.fill: parent; anchors.margins: 34; spacing: 14
                    Label { text: "Collection"; font.pixelSize: 28; font.bold: true }
                    Label { text: "Raw Input is registered only while a visible collection session is active. Raw events are written to Parquet."; wrapMode: Text.Wrap; Layout.fillWidth: true }
                    RowLayout {
                        Label { text: "Trials" }
                        SpinBox { id: trialCount; from: 1; to: 500; value: 20; editable: true; enabled: collectionPage.collector.state === "idle" }
                        Button {
                            text: "Launch fullscreen game"
                            enabled: collectionPage.collector.available && collectionPage.collector.state === "idle"
                            onClicked: {
                                collectionGame.requestedTrials = trialCount.value
                                collectionGame.showFullScreen()
                                collectionGame.requestActivate()
                            }
                        }
                    }
                    Label { text: collectionPage.collector.message; wrapMode: Text.Wrap; Layout.fillWidth: true }
                    Label { text: "Progress: " + collectionPage.collector.completedTrials + " / " + collectionPage.collector.plannedTrials }
                    Rectangle {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        color: "#e5e7eb"; radius: 8
                        Label { anchors.centerIn: parent; text: "Choose a trial count, then launch the fullscreen target game."; color: "#4b5563" }
                    }
                }
            }
            Item {
                id: reviewPage
                property var reviewer: appController.reviewController
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 34
                    spacing: 12
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Sessions and trial review"; font.pixelSize: 28; font.bold: true }
                        Item { Layout.fillWidth: true }
                        Button { text: "Refresh"; onClicked: reviewPage.reviewer.refresh() }
                    }
                    Label {
                        text: "Discarded sessions and trials are excluded from future dataset snapshots. Their raw files remain available and can be restored."
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                    Label { text: reviewPage.reviewer.message; color: "#4b5563"; Layout.fillWidth: true }
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 16
                        Rectangle {
                            Layout.preferredWidth: 330
                            Layout.fillHeight: true
                            color: "#ffffff"
                            border.color: "#d1d5db"
                            radius: 6
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                Label { text: "Recorded sessions"; font.bold: true; font.pixelSize: 17 }
                                Label { visible: reviewPage.reviewer.sessions.length === 0; text: "No sessions yet."; color: "#6b7280" }
                                ListView {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    clip: true
                                    spacing: 6
                                    model: reviewPage.reviewer.sessions
                                    delegate: Rectangle {
                                        required property var modelData
                                        width: ListView.view.width
                                        height: 88
                                        radius: 5
                                        color: modelData.review_disposition === "discarded" ? "#fee2e2" : "#f9fafb"
                                        border.color: "#d1d5db"
                                        Column {
                                            anchors.fill: parent
                                            anchors.margins: 8
                                            spacing: 3
                                            Text { text: modelData.display_name; font.bold: true; elide: Text.ElideRight; width: parent.width }
                                            Text { text: modelData.completed_trials + " completed / " + modelData.trial_count + " trials"; color: "#4b5563" }
                                            Row {
                                                spacing: 6
                                                Button { text: "Review"; height: 28; onClicked: reviewPage.reviewer.selectSession(modelData.id) }
                                                Button {
                                                    text: modelData.review_disposition === "discarded" ? "Restore" : "Discard"
                                                    height: 28
                                                    onClicked: modelData.review_disposition === "discarded" ? reviewPage.reviewer.retainSession(modelData.id) : reviewPage.reviewer.discardSession(modelData.id)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            color: "#ffffff"
                            border.color: "#d1d5db"
                            radius: 6
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                Label { text: reviewPage.reviewer.selectedSessionId === "" ? "Select a session" : "Trials"; font.bold: true; font.pixelSize: 17 }
                                Label { visible: reviewPage.reviewer.selectedSessionId !== "" && reviewPage.reviewer.trials.length === 0; text: "This session has no finalized trials."; color: "#6b7280" }
                                ListView {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 225
                                    Layout.fillHeight: false
                                    clip: true
                                    spacing: 6
                                    model: reviewPage.reviewer.trials
                                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                                    delegate: Rectangle {
                                        required property var modelData
                                        width: ListView.view.width
                                        height: 96
                                        radius: 5
                                        color: modelData.review_disposition === "discarded" ? "#fee2e2" : "#f9fafb"
                                        border.color: "#d1d5db"
                                        Row {
                                            anchors.fill: parent
                                            anchors.margins: 8
                                            spacing: 10
                                            Column {
                                                width: parent.width - trialActions.width - 10
                                                Text { text: "Trial " + modelData.id.slice(0, 8) + " — " + modelData.status; font.bold: true; width: parent.width; elide: Text.ElideRight }
                                                Text { text: "Target " + modelData.condition.target_x + ", " + modelData.condition.target_y + " · " + modelData.clicks.length + " clicks"; color: "#4b5563"; width: parent.width; elide: Text.ElideRight }
                                                Text { text: modelData.end_reason || "unfinished"; color: "#6b7280"; width: parent.width; elide: Text.ElideRight }
                                            }
                                            Column {
                                                id: trialActions
                                                anchors.verticalCenter: parent.verticalCenter
                                                spacing: 4
                                                width: 96
                                                Button { text: "View path"; width: parent.width; height: 30; onClicked: reviewPage.reviewer.selectTrial(modelData.id) }
                                                Button {
                                                    text: modelData.review_disposition === "discarded" ? "Restore" : "Discard"
                                                    width: parent.width
                                                    height: 30
                                                    onClicked: modelData.review_disposition === "discarded" ? reviewPage.reviewer.retainTrial(modelData.id) : reviewPage.reviewer.discardTrial(modelData.id)
                                                }
                                            }
                                        }
                                    }
                                }
                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    visible: reviewPage.reviewer.trajectory.raw_point_count > 0
                                    color: "#111827"
                                    radius: 5
                                    border.color: "#374151"
                                    Canvas {
                                        id: trajectoryCanvas
                                        anchors.fill: parent
                                        anchors.margins: 10
                                        property var trajectory: reviewPage.reviewer.trajectory
                                        onTrajectoryChanged: requestPaint()
                                        onWidthChanged: requestPaint()
                                        onHeightChanged: requestPaint()
                                        onPaint: {
                                            var ctx = getContext("2d")
                                            ctx.clearRect(0, 0, width, height)
                                            var data = trajectory
                                            if (!data || !data.points || data.points.length === 0) return
                                            var bounds = data.bounds
                                            var scaleX = width / Math.max(1, bounds.max_x - bounds.min_x)
                                            var scaleY = height / Math.max(1, bounds.max_y - bounds.min_y)
                                            var scale = Math.min(scaleX, scaleY)
                                            var offsetX = (width - (bounds.max_x - bounds.min_x) * scale) / 2
                                            var offsetY = (height - (bounds.max_y - bounds.min_y) * scale) / 2
                                            function px(point) { return offsetX + (point.x - bounds.min_x) * scale }
                                            function py(point) { return offsetY + (point.y - bounds.min_y) * scale }
                                            function rainbowColor(progress) {
                                                var hue = 275 * (1 - progress) / 60
                                                var sector = Math.floor(hue) % 6
                                                var fraction = hue - Math.floor(hue)
                                                var value = 0.96
                                                var saturation = 0.9
                                                var low = value * (1 - saturation)
                                                var falling = value * (1 - saturation * fraction)
                                                var rising = value * (1 - saturation * (1 - fraction))
                                                var red = 0
                                                var green = 0
                                                var blue = 0
                                                if (sector === 0) { red = value; green = rising; blue = low }
                                                else if (sector === 1) { red = falling; green = value; blue = low }
                                                else if (sector === 2) { red = low; green = value; blue = rising }
                                                else if (sector === 3) { red = low; green = falling; blue = value }
                                                else if (sector === 4) { red = rising; green = low; blue = value }
                                                else { red = value; green = low; blue = falling }
                                                return "rgb(" + Math.round(red * 255) + "," + Math.round(green * 255) + "," + Math.round(blue * 255) + ")"
                                            }
                                            // Put the target behind the path so the late red segment remains visible.
                                            ctx.fillStyle = "rgba(34, 197, 94, 0.20)"
                                            ctx.beginPath(); ctx.arc(px(data.target), py(data.target), Math.max(5, data.target.radius * scale), 0, Math.PI * 2); ctx.fill()
                                            ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2; ctx.stroke()
                                            ctx.lineWidth = 2
                                            var segmentCount = Math.max(1, data.points.length)
                                            var previous = data.start
                                            for (var i = 0; i < data.points.length; ++i) {
                                                // Violet at the first movement sample, red at the latest.
                                                ctx.strokeStyle = rainbowColor(i / Math.max(1, segmentCount - 1))
                                                ctx.beginPath()
                                                ctx.moveTo(px(previous), py(previous))
                                                ctx.lineTo(px(data.points[i]), py(data.points[i]))
                                                ctx.stroke()
                                                previous = data.points[i]
                                            }
                                            ctx.fillStyle = "#38bdf8"
                                            ctx.beginPath(); ctx.arc(px(data.start), py(data.start), 5, 0, Math.PI * 2); ctx.fill()
                                            var end = data.points[data.points.length - 1]
                                            ctx.fillStyle = "#facc15"
                                            ctx.beginPath(); ctx.arc(px(end), py(end), 4, 0, Math.PI * 2); ctx.fill()
                                        }
                                    }
                                    Label {
                                        anchors.left: parent.left; anchors.top: parent.top; anchors.margins: 12
                                        text: "Rainbow: early to late · Blue: start · Green: target · Yellow: final sample · " + reviewPage.reviewer.trajectory.raw_point_count + " raw points · " + reviewPage.reviewer.trajectory.duration_ms + " ms"
                                        color: "white"
                                        font.pixelSize: 12
                                        width: parent.width - 24
                                        wrapMode: Text.Wrap
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Window {
        id: collectionGame
        title: "MouseMotionLab Collection"
        color: "#111827"
        flags: Qt.FramelessWindowHint | Qt.Window
        property int requestedTrials: 20
        property bool launchPending: false

        onVisibleChanged: {
            if (visible && !launchPending && collectionPage.collector.state === "idle") {
                launchPending = true
                gameStartTimer.restart()
            }
        }
        onClosing: function(close) {
            if (collectionPage.collector.state === "active") {
                close.accepted = false
                appController.stopCollection()
            }
        }

        Timer {
            id: gameStartTimer
            interval: 120
            repeat: false
            onTriggered: {
                collectionGame.launchPending = false
                appController.startCollection(collectionGame, collectionGame.requestedTrials, gameCanvas.x, gameCanvas.y, gameCanvas.width, gameCanvas.height)
            }
        }

        Connections {
            target: collectionPage.collector
            function onStateChanged() {
                if (collectionGame.visible && collectionPage.collector.state === "idle")
                    collectionGame.close()
            }
        }

        Rectangle {
            id: gameCanvas
            anchors.fill: parent
            color: "#111827"
            focus: true
            Keys.onEscapePressed: appController.stopCollection()

            Label {
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 28
                text: "Click the green target. Esc stops and saves the session."
                color: "#d1d5db"
                font.pixelSize: 16
            }
            Label {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.margins: 28
                text: "Trial " + (collectionPage.collector.completedTrials + 1) + " / " + collectionPage.collector.plannedTrials
                color: "#d1d5db"
                font.pixelSize: 16
            }
            Button {
                anchors.top: parent.top
                anchors.right: parent.right
                anchors.margins: 22
                text: "Stop and save"
                onClicked: appController.stopCollection()
            }
            Label {
                anchors.centerIn: parent
                visible: !collectionPage.collector.targetVisible
                text: collectionPage.collector.state === "active" ? "Prepare for the next target" : "Starting collection…"
                color: "#d1d5db"
                font.pixelSize: 18
            }
            Rectangle {
                visible: collectionPage.collector.targetVisible
                width: collectionPage.collector.targetRadius * 2
                height: width
                radius: width / 2
                x: collectionPage.collector.targetX - collectionPage.collector.targetRadius
                y: collectionPage.collector.targetY - collectionPage.collector.targetRadius
                color: "#22c55e"
                border.color: "white"
                border.width: 3
            }
        }
    }
}
