import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page
    objectName: "experimentsPage"
    property var trainingController
    property var datasetController
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 28; spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Conditional-flow experiments"; font.pixelSize: 28; font.bold: true }
            Item { Layout.fillWidth: true }
            Button { text: "Refresh"; onClicked: { page.datasetController.refresh(); page.trainingController.refresh() } }
        }
        Label {
            text: "Training runs in a separate worker. Each experiment is tied to an immutable dataset and records checkpoints, losses, seed, environment, and its published model."
            wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563"
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Flow training data" }
            ComboBox {
                id: runChoice
                objectName: "flowTrainingDataChoice"
                Layout.fillWidth: true
                Layout.minimumWidth: 180
                Layout.maximumWidth: 390
                model: page.datasetController.preprocessingRuns; textRole: "display_name"
                displayText: currentIndex < 0 ? "No processed datasets" : model[currentIndex].display_name
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: displayText
                ToolTip.delay: 400
                ToolTip.timeout: 10000
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Item { Layout.fillWidth: true }
            Button {
                text: "Train small"
                enabled: runChoice.currentIndex >= 0 && runChoice.model[runChoice.currentIndex].status === "completed"
                onClicked: page.trainingController.startTraining(runChoice.model[runChoice.currentIndex].id, "small")
            }
            Button {
                objectName: "trainStandardButton"
                text: "Train standard"
                enabled: runChoice.currentIndex >= 0 && runChoice.model[runChoice.currentIndex].status === "completed"
                onClicked: page.trainingController.startTraining(runChoice.model[runChoice.currentIndex].id, "standard")
            }
        }
        Label { text: page.trainingController.message; wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563" }
        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true; color: "white"; radius: 6; border.color: "#d1d5db"
            Label { anchors.centerIn: parent; visible: page.trainingController.experiments.length === 0; text: "No conditional-flow experiments yet."; color: "#6b7280" }
            ListView {
                anchors.fill: parent; anchors.margins: 12; clip: true; spacing: 7
                model: page.trainingController.experiments
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view.width; height: 82; radius: 5; color: "#f9fafb"; border.color: "#d1d5db"
                    Column {
                        anchors.fill: parent; anchors.margins: 9; spacing: 3
                        Text { text: modelData.name + " — " + modelData.status; font.bold: true }
                        Text { text: "Epoch " + modelData.latest_epoch + " · seed " + modelData.random_seed + " · " + modelData.preprocessing_run_id.slice(0, 8); color: "#4b5563" }
                        Text {
                            text: modelData.best_validation_loss === null ? (modelData.error || "Waiting for metrics") : "Best validation loss " + Number(modelData.best_validation_loss).toFixed(5)
                            color: modelData.status === "failed" ? "#b91c1c" : "#047857"; elide: Text.ElideRight; width: parent.width
                        }
                    }
                }
            }
        }
    }
}
