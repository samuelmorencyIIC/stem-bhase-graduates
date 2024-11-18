window.addEventListener("message", function(event) {
    if (event.data.type && event.data.type === "map_click") {
        var cmapuid = event.data.data;
        var hiddenDiv = document.getElementById("map-click");
        if (hiddenDiv) {
            hiddenDiv.innerText = cmapuid;
            hiddenDiv.dispatchEvent(new Event('change'));
        }
    }
}, false);
