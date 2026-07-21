import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Mirrors ASPECT_RATIOS from __init__.py — keep these two in sync if you
// add/remove/rename presets on the Python side.
const ASPECT_RATIOS = {
    "1:1 (Perfect Square)": [1, 1],
    "2:3 (Classic Portrait)": [2, 3],
    "3:4 (Golden Ratio)": [3, 4],
    "3:5 (Elegant Vertical)": [3, 5],
    "4:5 (Artistic Frame)": [4, 5],
    "5:7 (Balanced Portrait)": [5, 7],
    "5:8 (Tall Portrait)": [5, 8],
    "7:9 (Modern Portrait)": [7, 9],
    "9:16 (Slim Vertical)": [9, 16],
    "9:19 (Tall Slim)": [9, 19],
    "9:21 (Ultra Tall)": [9, 21],
    "9:32 (Skyline)": [9, 32],
    "3:2 (Golden Landscape)": [3, 2],
    "4:3 (Classic Landscape)": [4, 3],
    "5:3 (Wide Horizon)": [5, 3],
    "5:4 (Balanced Frame)": [5, 4],
    "7:5 (Elegant Landscape)": [7, 5],
    "8:5 (Cinematic View)": [8, 5],
    "9:7 (Artful Horizon)": [9, 7],
    "16:9 (Panorama)": [16, 9],
    "19:9 (Cinematic Ultrawide)": [19, 9],
    "21:9 (Epic Ultrawide)": [21, 9],
    "32:9 (Extreme Ultrawide)": [32, 9],
    "custom": null,
};

function parseCustomRatio(str) {
    try {
        const [wStr, hStr] = String(str).split(":");
        const w = parseFloat(wStr.trim());
        const h = parseFloat(hStr.trim());
        if (!(w > 0) || !(h > 0)) throw new Error("invalid ratio");
        return [w, h];
    } catch {
        return [1, 1];
    }
}

// Guarantees the custom_ratio widget always displays a valid "W:H" string.
// Falls back to "1:1" for anything malformed — including a leftover value
// from before custom_ratio_w/custom_ratio_h were merged into one field.
const CUSTOM_RATIO_PATTERN = /^\s*\d+(\.\d+)?\s*:\s*\d+(\.\d+)?\s*$/;

function normalizeCustomRatio(widget) {
    if (!widget) return;
    if (typeof widget.value !== "string" || !CUSTOM_RATIO_PATTERN.test(widget.value)) {
        widget.value = "1:1";
    }
}

function roundToMultiple(value, multiple) {
    return Math.max(multiple, Math.round(value / multiple) * multiple);
}

function resolveDimensions(ratioW, ratioH, megapixels, snap) {
    const targetPixels = megapixels * 1_000_000;
    const rawW = Math.sqrt(targetPixels * (ratioW / ratioH));
    const rawH = rawW * (ratioH / ratioW);
    return [roundToMultiple(rawW, snap), roundToMultiple(rawH, snap)];
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// --- Remember last-used values for new nodes ------------------------------
// ComfyUI has no built-in "start new nodes from what I used last time"
// behavior — every freshly placed node starts from the hardcoded defaults
// in INPUT_TYPES. We replicate that convenience ourselves via localStorage.
// Saved workflows are unaffected: onConfigure() runs after onNodeCreated()
// and overwrites these with whatever was actually saved in the workflow.
const LAST_VALUES_STORAGE_KEY = "FxSmartLatentImage.lastValues";
const REMEMBERED_WIDGET_NAMES = ["aspect_ratio", "megapixels", "snap_to_multiple_of"];

function loadLastValues() {
    try {
        return JSON.parse(localStorage.getItem(LAST_VALUES_STORAGE_KEY)) || {};
    } catch {
        return {};
    }
}

function saveLastValue(name, value) {
    try {
        const values = loadLastValues();
        values[name] = value;
        localStorage.setItem(LAST_VALUES_STORAGE_KEY, JSON.stringify(values));
    } catch (err) {
        console.warn("[Fx Smart Latent Image] failed to save last-used value:", err);
    }
}

// Recomputes width/height client-side (mirroring the Python math) so the
// node shows the resulting resolution immediately as you change settings,
// without needing to run the workflow first. This widget is display-only:
// it has no corresponding output socket, so nothing here is ever sent to
// other nodes — it only updates what's drawn on this node itself.
function updateDisplay(node) {
    const aspectWidget = getWidget(node, "aspect_ratio");
    const megapixelsWidget = getWidget(node, "megapixels");
    const snapWidget = getWidget(node, "snap_to_multiple_of");
    const customRatioWidget = getWidget(node, "custom_ratio");
    const displayWidget = getWidget(node, "resolution_display");
    if (!aspectWidget || !megapixelsWidget || !snapWidget || !displayWidget) return;

    const preset = ASPECT_RATIOS[aspectWidget.value];
    const [ratioW, ratioH] = preset ? preset : parseCustomRatio(customRatioWidget?.value ?? "1:1");

    const megapixels = parseFloat(megapixelsWidget.value);
    const snap = parseInt(snapWidget.value, 10);
    const [width, height] = resolveDimensions(ratioW, ratioH, megapixels, snap);

    displayWidget.value = `${width} x ${height}  (${((width * height) / 1_000_000).toFixed(2)} MP)`;
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Comfy.EmptyLatentResolutionCalc",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "EmptyLatentResolutionCalc") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                try {
                    // create the read-only resolution display widget
                    let widget = getWidget(this, "resolution_display");
                    if (!widget) {
                        widget = ComfyWidgets["STRING"](
                            this,
                            "resolution_display",
                            ["STRING", { multiline: false }],
                            app
                        ).widget;
                        // Best-effort "read only": works whether this widget is
                        // a plain canvas-drawn text widget or a DOM element —
                        // never assume inputEl exists.
                        widget.disabled = true;
                        if (widget.inputEl) {
                            widget.inputEl.readOnly = true;
                            widget.inputEl.style.opacity = 0.7;
                            widget.inputEl.style.textAlign = "center";
                        }
                    }

                    // Apply remembered last-used values (only affects
                    // freshly placed nodes — a loaded workflow's onConfigure
                    // runs next and overwrites these with the saved values).
                    const lastValues = loadLastValues();
                    for (const name of REMEMBERED_WIDGET_NAMES) {
                        const w = getWidget(this, name);
                        if (w && lastValues[name] !== undefined) {
                            w.value = lastValues[name];
                        }
                    }

                    // Ensure custom_ratio always shows a valid "1:1"-style
                    // value (guards against any leftover/corrupted value).
                    normalizeCustomRatio(getWidget(this, "custom_ratio"));

                    // live-update the preview whenever a relevant widget changes
                    for (const name of ["aspect_ratio", "megapixels", "snap_to_multiple_of", "custom_ratio"]) {
                        const w = getWidget(this, name);
                        if (!w) continue;
                        const origCallback = w.callback;
                        w.callback = (...args) => {
                            const r = origCallback?.apply(w, args);
                            if (REMEMBERED_WIDGET_NAMES.includes(name)) {
                                saveLastValue(name, w.value);
                            }
                            try {
                                updateDisplay(this);
                            } catch (err) {
                                console.warn("[Fx Smart Latent Image] live update failed:", err);
                            }
                            return r;
                        };
                    }

                    // initial calculation as soon as the node is placed
                    setTimeout(() => {
                        try {
                            updateDisplay(this);
                        } catch (err) {
                            console.warn("[Fx Smart Latent Image] initial display update failed:", err);
                        }
                    }, 0);
                } catch (err) {
                    console.warn("[Fx Smart Latent Image] onNodeCreated failed:", err);
                }
            };

            // also recompute when a saved workflow is loaded
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                setTimeout(() => {
                    try {
                        normalizeCustomRatio(getWidget(this, "custom_ratio"));
                        updateDisplay(this);
                    } catch (err) {
                        console.warn("[Fx Smart Latent Image] onConfigure display update failed:", err);
                    }
                }, 0);
                return r;
            };
        }
    },
});
