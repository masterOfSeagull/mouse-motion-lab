import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page
    objectName: "generatorPage"
    focus: true
    Keys.onEscapePressed: appController.playbackController.abortFromUi()
    property var generatorController
    property var datasetController

    function latestModelIndex(modelType) {
        for (var index = 0; index < page.generatorController.models.length; ++index) {
            if (page.generatorController.models[index].model_type === modelType)
                return index
        }
        return -1
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 28
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Trajectory generator"; font.pixelSize: 28; font.bold: true }
            Item { Layout.fillWidth: true }
            Button { text: "Refresh"; onClicked: { page.datasetController.refresh(); page.generatorController.refresh() } }
        }
        Label {
            text: "The processed-dataset selector below is only for building retrieval and PCA models. Choose any existing generator, including conditional flow, from the separate model selector. Preview never injects input."
            wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563"
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Retrieval/PCA training data" }
            ComboBox {
                id: runChoice
                objectName: "generatorTrainingDataChoice"
                Layout.fillWidth: true
                Layout.minimumWidth: 180
                Layout.maximumWidth: 390
                model: page.datasetController.preprocessingRuns
                textRole: "display_name"
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
                text: "Build retrieval"
                enabled: runChoice.currentIndex >= 0 && runChoice.model[runChoice.currentIndex].status === "completed"
                onClicked: page.generatorController.buildBaseline(runChoice.model[runChoice.currentIndex].id, "retrieval")
            }
            Button {
                objectName: "buildPcaButton"
                text: "Build PCA mixture"
                enabled: runChoice.currentIndex >= 0 && runChoice.model[runChoice.currentIndex].status === "completed"
                onClicked: page.generatorController.buildBaseline(runChoice.model[runChoice.currentIndex].id, "pca_mixture")
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Generator model" }
            ComboBox {
                id: modelChoice
                objectName: "generatorModelChoice"
                Layout.fillWidth: true
                Layout.minimumWidth: 220
                Layout.maximumWidth: 440
                model: page.generatorController.models
                textRole: "display_name"
                displayText: currentIndex < 0 ? "No ready generators" : currentText
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: displayText
                ToolTip.delay: 400
                ToolTip.timeout: 10000
            }
            Button {
                objectName: "useTrainedModelButton"
                text: "Use trained model"
                enabled: page.latestModelIndex("conditional_flow") >= 0
                onClicked: modelChoice.currentIndex = page.latestModelIndex("conditional_flow")
                hoverEnabled: true
                ToolTip.visible: hovered
                ToolTip.text: "Switch to the newest trained conditional-flow model"
                ToolTip.delay: 400
                ToolTip.timeout: 10000
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Item { Layout.fillWidth: true }
            Label { text: "Seed (-1 = random)" }
            SpinBox { id: seed; objectName: "generationSeed"; from: -1; to: 999999999; value: 42; editable: true }
            Label { text: "Radius" }
            SpinBox { id: targetRadius; from: 4; to: 100; value: 28; editable: true }
            Button {
                objectName: "generateButton"
                text: "Generate"
                enabled: modelChoice.currentIndex >= 0 && page.generatorController.models.length > 0
                onClicked: page.generatorController.generate(
                    modelChoice.model[modelChoice.currentIndex].id,
                    startX.value, startY.value, targetX.value, targetY.value, targetRadius.value, seed.value
                )
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Start X" }
            SpinBox { id: startX; from: -10000; to: 10000; value: 120; editable: true }
            Label { text: "Y" }
            SpinBox { id: startY; from: -10000; to: 10000; value: 260; editable: true }
            Label { text: "Target X" }
            SpinBox { id: targetX; from: -10000; to: 10000; value: 620; editable: true }
            Label { text: "Y" }
            SpinBox { id: targetY; from: -10000; to: 10000; value: 180; editable: true }
        }
        Label { text: page.generatorController.message; wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#4b5563" }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label {
                text: "Playback: " + appController.playbackController.state
                color: appController.playbackController.state === "armed" ? "#b45309" : appController.playbackController.state === "playing" ? "#b91c1c" : "#4b5563"
                font.bold: appController.playbackController.state !== "disarmed"
            }
            Button {
                text: "Arm local playback"
                enabled: appController.playbackController.state === "disarmed" && page.generatorController.trajectory.points !== undefined
                onClicked: { page.forceActiveFocus(); appController.playbackController.arm() }
            }
            Button {
                text: "Start once"
                enabled: appController.playbackController.state === "armed"
                onClicked: { page.forceActiveFocus(); appController.startPlayback() }
            }
            Button {
                text: "Abort"
                enabled: appController.playbackController.state === "playing"
                onClicked: appController.playbackController.abortFromUi()
            }
            Label { text: appController.playbackController.message; Layout.fillWidth: true; wrapMode: Text.Wrap; color: "#4b5563" }
        }
        Label {
            visible: modelChoice.currentIndex >= 0 && modelChoice.model[modelChoice.currentIndex].validation_summary !== undefined
            text: modelChoice.currentIndex < 0 || modelChoice.model[modelChoice.currentIndex].validation_summary === undefined ?
                      "" : modelChoice.model[modelChoice.currentIndex].validation_summary
            color: "#047857"
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Label { text: "Trajectory viewport"; font.bold: true }
            Label {
                text: {
                    var desktop = page.generatorController.trajectory.desktop
                    return desktop === undefined ? "Generate to inspect the desktop" :
                           Number(desktop.width).toFixed(0) + " x " + Number(desktop.height).toFixed(0) + " desktop"
                }
                color: "#4b5563"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                elide: Text.ElideRight
            }
            Label { text: "Zoom" }
            Slider {
                id: zoomSlider
                objectName: "trajectoryZoom"
                from: 1; to: 4; stepSize: 0.25; value: 1
                Layout.preferredWidth: 150
                onMoved: trajectoryViewport.returnToBounds()
            }
            Label { text: Math.round(zoomSlider.value * 100) + "%"; Layout.preferredWidth: 45 }
            Button {
                objectName: "fitTrajectoryButton"
                text: "Fit"
                onClicked: {
                    zoomSlider.value = 1
                    trajectoryViewport.returnToBounds()
                }
            }
        }
        Rectangle {
            id: preview
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 80
            color: "#030712"; radius: 6; border.color: "#374151"
            clip: true
            readonly property var trajectory: page.generatorController.trajectory
            readonly property var desktop: trajectory.desktop === undefined ?
                                               ({"left": 0, "top": 0, "width": 1920, "height": 1080}) :
                                               trajectory.desktop
            readonly property real desktopWidth: Math.max(1, Number(desktop.width))
            readonly property real desktopHeight: Math.max(1, Number(desktop.height))
            readonly property real fitScale: Math.max(
                0.01,
                Math.min(
                    Math.max(1, trajectoryViewport.width - 24) / desktopWidth,
                    Math.max(1, trajectoryViewport.height - 24) / desktopHeight
                )
            )

            Flickable {
                id: trajectoryViewport
                objectName: "trajectoryViewport"
                anchors.fill: parent
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                contentWidth: contentSurface.width
                contentHeight: contentSurface.height
                interactive: zoomSlider.value > 1
                ScrollBar.horizontal: ScrollBar {
                    objectName: "trajectoryHorizontalScroll"
                    policy: zoomSlider.value > 1 ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                    active: zoomSlider.value > 1
                    contentItem: Rectangle {
                        implicitHeight: 9
                        radius: 4
                        color: parent.pressed ? "#e2e8f0" : "#94a3b8"
                    }
                    background: Rectangle { color: "#1f2937" }
                }
                ScrollBar.vertical: ScrollBar {
                    objectName: "trajectoryVerticalScroll"
                    policy: zoomSlider.value > 1 ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                    active: zoomSlider.value > 1
                    contentItem: Rectangle {
                        implicitWidth: 9
                        radius: 4
                        color: parent.pressed ? "#e2e8f0" : "#94a3b8"
                    }
                    background: Rectangle { color: "#1f2937" }
                }

                Item {
                    id: contentSurface
                    width: Math.max(trajectoryViewport.width, screenFrame.width + 24)
                    height: Math.max(trajectoryViewport.height, screenFrame.height + 24)

                    Rectangle {
                        id: screenFrame
                        objectName: "trajectoryScreenFrame"
                        x: (contentSurface.width - width) / 2
                        y: (contentSurface.height - height) / 2
                        width: preview.desktopWidth * preview.fitScale * zoomSlider.value
                        height: preview.desktopHeight * preview.fitScale * zoomSlider.value
                        color: "#111827"
                        border.color: "#64748b"
                        border.width: 1
                        clip: true

                        Canvas {
                            id: pathCanvas
                            anchors.fill: parent
                            property var trajectory: preview.trajectory
                            onTrajectoryChanged: requestPaint()
                            onWidthChanged: requestPaint()
                            onHeightChanged: requestPaint()
                            onPaint: {
                                var ctx = getContext("2d")
                                ctx.clearRect(0, 0, width, height)
                                var data = trajectory
                                if (!data || !data.points || data.points.length === 0) return
                                var scaleX = width / preview.desktopWidth
                                var scaleY = height / preview.desktopHeight
                                function px(point) { return (point.x - Number(preview.desktop.left)) * scaleX }
                                function py(point) { return (point.y - Number(preview.desktop.top)) * scaleY }
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
                                ctx.fillStyle = "#ffffff"
                                ctx.beginPath(); ctx.arc(px(data.target), py(data.target), data.target.radius * Math.min(scaleX, scaleY), 0, Math.PI * 2); ctx.fill()
                                for (var i = 1; i < data.points.length; ++i) {
                                    var progress = (i - 1) / Math.max(1, data.points.length - 2)
                                    ctx.strokeStyle = rainbowColor(progress)
                                    ctx.lineWidth = 2
                                    ctx.beginPath(); ctx.moveTo(px(data.points[i - 1]), py(data.points[i - 1]))
                                    ctx.lineTo(px(data.points[i]), py(data.points[i])); ctx.stroke()
                                }
                                ctx.fillStyle = "#38bdf8"; ctx.beginPath(); ctx.arc(px(data.start), py(data.start), 5, 0, Math.PI * 2); ctx.fill()
                                var end = data.points[data.points.length - 1]
                                ctx.fillStyle = "#facc15"; ctx.beginPath(); ctx.arc(px(end), py(end), 4, 0, Math.PI * 2); ctx.fill()
                            }
                        }
                        Label {
                            anchors.left: parent.left; anchors.top: parent.top; anchors.margins: 10
                            width: parent.width - 20
                            color: "white"; font.pixelSize: 12
                            elide: Text.ElideRight
                            text: {
                                var data = preview.trajectory
                                if (!data || !data.points) return "Full desktop viewport - Rainbow: early to late - Blue: start - White: target - Yellow: final sample"
                                return Number(data.duration_ms).toFixed(1) + " ms - path " + Number(data.path_length).toFixed(1) +
                                       " px - peak " + Number(data.peak_speed).toFixed(0) + " px/s - distance score " +
                                       Number(data.condition_distance).toFixed(2) + " - seed " + data.seed
                                       + (data.source_index === undefined || data.source_index < 0 ? "" : " - source sample " + data.source_index)
                                       + " - " + data.condition_class.split("_").join(" ")
                            }
                        }
                        Rectangle {
                            visible: appController.playbackController.state === "playing"
                            x: (appController.playbackController.x - Number(preview.desktop.left)) * screenFrame.width / preview.desktopWidth - width / 2
                            y: (appController.playbackController.y - Number(preview.desktop.top)) * screenFrame.height / preview.desktopHeight - height / 2
                            width: 14; height: 14; radius: 7
                            color: "#ffffff"; border.color: "#ef4444"; border.width: 3
                        }
                    }
                }
            }
        }
    }
}
