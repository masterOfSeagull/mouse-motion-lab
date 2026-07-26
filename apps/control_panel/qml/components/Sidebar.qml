import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: sidebar
    property int currentPage: 0
    signal pageSelected(int page)

    color: "#1f2937"
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        Label { text: "MouseMotionLab"; font.bold: true; font.pixelSize: 19; color: "white" }
        Label { text: "Milestones 1-8 complete"; color: "#9ca3af" }
        Button { text: "Dashboard"; onClicked: sidebar.pageSelected(0); Layout.fillWidth: true }
        Button { text: "Diagnostics"; onClicked: sidebar.pageSelected(1); Layout.fillWidth: true }
        Button { text: "Jobs"; onClicked: sidebar.pageSelected(2); Layout.fillWidth: true }
        Button { text: "Collection"; onClicked: sidebar.pageSelected(3); Layout.fillWidth: true }
        Button { text: "Sessions"; onClicked: sidebar.pageSelected(4); Layout.fillWidth: true }
        Button { text: "Datasets"; onClicked: sidebar.pageSelected(5); Layout.fillWidth: true }
        Button { text: "Generator"; onClicked: sidebar.pageSelected(6); Layout.fillWidth: true }
        Button { text: "Experiments"; onClicked: sidebar.pageSelected(7); Layout.fillWidth: true }
        Button { text: "Models"; onClicked: sidebar.pageSelected(8); Layout.fillWidth: true }
        Item { Layout.fillHeight: true }
        Label { text: "External playback disabled"; wrapMode: Text.Wrap; color: "#9ca3af"; Layout.fillWidth: true }
    }
}
