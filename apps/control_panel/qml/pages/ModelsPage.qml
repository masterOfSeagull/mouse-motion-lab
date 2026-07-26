import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page
    property var registryController
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 28; spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Label { text: "Validated model registry"; font.pixelSize: 28; font.bold: true }
            Item { Layout.fillWidth: true }
            Button { text: "Refresh"; onClicked: page.registryController.refresh() }
        }
        Label { text: "Only models with an intact passing report can be promoted. At most one model is active."; wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563" }
        Label { text: page.registryController.message; wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563" }
        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true; color: "white"; radius: 6; border.color: "#d1d5db"
            ListView {
                anchors.fill: parent; anchors.margins: 12; clip: true; spacing: 8
                model: page.registryController.models
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                delegate: Rectangle {
                    required property var modelData
                    width: ListView.view.width; height: 112; radius: 5
                    color: modelData.lifecycle === "active" ? "#dcfce7" : "#f9fafb"
                    border.color: modelData.lifecycle === "active" ? "#16a34a" : "#d1d5db"
                    RowLayout {
                        anchors.fill: parent; anchors.margins: 10
                        ColumnLayout {
                            Layout.fillWidth: true; spacing: 3
                            Text { text: modelData.name + " — " + modelData.lifecycle; font.bold: true }
                            Text { text: modelData.model_type + " · snapshot " + modelData.dataset_snapshot_id.slice(0, 8) + " · " + modelData.id.slice(0, 8); color: "#4b5563" }
                            Text {
                                text: modelData.validation_error ? modelData.validation_error :
                                      "Duration W1 " + Number(modelData.duration_wasserstein_ns / 1000000).toFixed(2) + " ms · path W1 " + Number(modelData.path_wasserstein).toFixed(3) + " · OOD " + modelData.ood_count
                                color: modelData.validation_error ? "#b91c1c" : "#047857"
                            }
                        }
                        ColumnLayout {
                            Button {
                                text: modelData.lifecycle === "active" ? "Active" : "Promote"
                                enabled: !modelData.validation_error && modelData.lifecycle !== "active" && modelData.lifecycle !== "deprecated"
                                onClicked: page.registryController.promote(modelData.id)
                            }
                            Button {
                                text: "Export ONNX"
                                enabled: modelData.model_type === "conditional_flow" && !modelData.validation_error
                                onClicked: page.registryController.exportModel(modelData.id)
                            }
                        }
                    }
                }
            }
        }
    }
}
