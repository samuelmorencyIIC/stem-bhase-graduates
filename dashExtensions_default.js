window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature) {
                return feature.properties.style; // Set style for each feature
            }

            ,
        function1: function(feature, layer) {
                if (feature.properties && feature.properties.tooltip) {
                    layer.bindTooltip(feature.properties.tooltip); // Add tooltip to each feature
                }
            }

            ,
        function2: function(e, ctx) {
            e.originalEvent._stopped = true; // Prevent event from bubbling up to the map
            ctx.setProps({
                clickedFeature: e.sourceTarget.feature.properties.DGUID
            });
        }

    }
});