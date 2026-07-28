import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: datasetPage
    property var datasetController
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 34
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Dataset snapshots"; font.pixelSize: 28; font.bold: true }
            Item { Layout.fillWidth: true }
            Button { text: "Refresh"; onClicked: datasetPage.datasetController.refresh() }
        }
        Label {
            text: "Snapshots are immutable manifests. Build uses retained current-protocol sessions only; each session stays entirely in one split."
            wrapMode: Text.Wrap
            Layout.fillWidth: true
        }
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Name" }
            TextField { id: snapshotName; Layout.fillWidth: true; text: "Dataset snapshot"; selectByMouse: true }
            Button { text: "Build snapshot"; onClicked: datasetPage.datasetController.buildSnapshot(snapshotName.text) }
        }
        Label { text: datasetPage.datasetController.message; color: "#4b5563"; wrapMode: Text.Wrap; Layout.fillWidth: true }
        Label {
            Layout.fillWidth: true
            visible: datasetPage.datasetController.preprocessingRuns.length > 0
            text: {
                var run = datasetPage.datasetController.preprocessingRuns[0]
                var resampling = run.resampling_max_error === undefined || run.resampling_max_error === null ? "" : " • resampling error " + Number(run.resampling_max_error).toFixed(4)
                return "Latest processed dataset: " + run.display_name + " | " + run.processed_trial_count + " processed | " + run.skipped_trial_count + " skipped" + resampling
            }
            color: datasetPage.datasetController.preprocessingRuns.length && datasetPage.datasetController.preprocessingRuns[0].status === "completed" ? "#047857" : "#b45309"
            wrapMode: Text.Wrap
        }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#ffffff"
            border.color: "#d1d5db"
            radius: 6
            Label { anchors.centerIn: parent; visible: datasetPage.datasetController.snapshots.length === 0; text: "No snapshots yet. Retained completed sessions with verified raw files are eligible."; color: "#6b7280"; width: parent.width - 30; wrapMode: Text.Wrap; horizontalAlignment: Text.AlignHCenter }
            ListView {
                anchors.fill: parent
                anchors.margins: 12
                clip: true
                spacing: 6
                model: datasetPage.datasetController.snapshots
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view.width
                    height: 106
                    radius: 5
                    color: "#f9fafb"
                    border.color: "#d1d5db"
                    Column {
                        anchors.fill: parent
                        anchors.margins: 9
                        spacing: 3
                        Button { text: "Preprocess"; onClicked: datasetPage.datasetController.preprocessSnapshot(modelData.id) }
                        Text { text: modelData.name + " — " + modelData.status; font.bold: true; width: parent.width; elide: Text.ElideRight }
                        Text { text: modelData.trial_count + " trials · " + modelData.session_count + " sessions · " + modelData.id.slice(0, 8); color: "#4b5563" }
                        Text { text: modelData.warnings.length ? modelData.warnings.join(" ") : "Session-held-out split ready"; color: modelData.warnings.length ? "#b45309" : "#047857"; width: parent.width; elide: Text.ElideRight }
                    }
                }
            }
        }
    }
}
