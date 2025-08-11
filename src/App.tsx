import React, { useRef, useState, useEffect } from "react";
import { Stage, Layer, Image as KonvaImage, Transformer } from "react-konva";

// ---- Image processing helpers (copied & adapted) ----
function gaussianKernel1D(sigma) {
  if (sigma <= 0) return { kernel: new Float32Array([1]), radius: 0 };
  const radius = Math.ceil(sigma * 3);
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  const sigma2 = sigma * sigma;
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma2));
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return { kernel, radius };
}

function separableGaussianBlurFloat(srcFloat, w, h, sigma) {
  const { kernel, radius } = gaussianKernel1D(sigma);
  const tmp = new Float32Array(srcFloat.length);
  const out = new Float32Array(srcFloat.length);
  const channels = 4;

  // horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 4; c++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const xx = Math.min(w - 1, Math.max(0, x + k));
          const idx = (y * w + xx) * channels + c;
          val += srcFloat[idx] * kernel[k + radius];
        }
        tmp[(y * w + x) * channels + c] = val;
      }
    }
  }

  // vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 4; c++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const yy = Math.min(h - 1, Math.max(0, y + k));
          const idx = (yy * w + x) * channels + c;
          val += tmp[idx] * kernel[k + radius];
        }
        out[(y * w + x) * channels + c] = val;
      }
    }
  }
  return out;
}

function toFloat32(srcU8) {
  const f = new Float32Array(srcU8.length);
  for (let i = 0; i < srcU8.length; i++) f[i] = srcU8[i] / 255;
  return f;
}

function floatToU8Clamped(floatArr, origU8) {
  const u8 = new Uint8ClampedArray(floatArr.length);
  for (let i = 0; i < floatArr.length; i += 4) {
    u8[i] = Math.round(Math.max(0, Math.min(1, floatArr[i])) * 255);
    u8[i + 1] = Math.round(Math.max(0, Math.min(1, floatArr[i + 1])) * 255);
    u8[i + 2] = Math.round(Math.max(0, Math.min(1, floatArr[i + 2])) * 255);
    u8[i + 3] = origU8[i + 3];
  }
  return u8;
}

function unsharpMaskImageData(
  imageData,
  amount = 5.0,
  radius = 20,
  threshold = 5
) {
  const w = imageData.width,
    h = imageData.height;
  const srcU8 = imageData.data;
  const srcF = toFloat32(srcU8);
  const blurredF = separableGaussianBlurFloat(srcF, w, h, radius);
  const outF = new Float32Array(srcF.length);

  for (let i = 0; i < srcF.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const s = srcF[i + c];
      const b = blurredF[i + c];
      let m = s - b;
      if (Math.abs(m) * 255 < threshold) m = 0;
      let sharp = s + amount * m;
      if (sharp < 0) sharp = 0;
      if (sharp > 1) sharp = 1;
      outF[i + c] = sharp;
    }
    outF[i + 3] = srcF[i + 3];
  }
  const outU8 = floatToU8Clamped(outF, srcU8);
  return new ImageData(outU8, w, h);
}

// overlay stroke onto a base image (returns a new ImageData)
function overlayStrokeOnImage(
  baseImageData,
  maskSourceImageData,
  strokeSize,
  strokeColorHex
) {
  const w = baseImageData.width;
  const h = baseImageData.height;
  const base = new Uint8ClampedArray(baseImageData.data); // copy

  // build mask from maskSourceImageData.alpha (if none, fall back to base)
  const maskSrc = maskSourceImageData || baseImageData;
  const mask = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < maskSrc.data.length; i += 4, p++) {
    mask[p] = maskSrc.data[i + 3] > 0 ? 1 : 0;
  }

  // parse color
  const r = parseInt(strokeColorHex.substr(1, 2), 16);
  const g = parseInt(strokeColorHex.substr(3, 2), 16);
  const b = parseInt(strokeColorHex.substr(5, 2), 16);
  const a = 255;

  const s = Math.max(0, Math.floor(strokeSize));
  if (s === 0) return new ImageData(base, w, h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (mask[idx] === 1) continue; // inside object -> skip

      let touching = false;
      // square neighborhood; could be optimized
      for (let ny = Math.max(0, y - s); ny <= Math.min(h - 1, y + s); ny++) {
        for (let nx = Math.max(0, x - s); nx <= Math.min(w - 1, x + s); nx++) {
          if (mask[ny * w + nx] === 1) {
            touching = true;
            break;
          }
        }
        if (touching) break;
      }

      if (touching) {
        const p = idx * 4;
        base[p] = r;
        base[p + 1] = g;
        base[p + 2] = b;
        base[p + 3] = a;
      }
    }
  }

  return new ImageData(base, w, h);
}

// ===================== Component React =====================
export default function UnsharpMaskApp() {
  const [imageObj, setImageObj] = useState(null);
  const [originalImageData, setOriginalImageData] = useState(null); // never changed
  const [filteredImageData, setFilteredImageData] = useState(null); // base image (filters applied, no stroke)
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [isSelected, setIsSelected] = useState(true);

  const [amount, setAmount] = useState(5.0);
  const [radius, setRadius] = useState(20);
  const [threshold, setThreshold] = useState(5);
  const [strokeSize, setStrokeSize] = useState(0);
  const [strokeColor, setStrokeColor] = useState("#000000");

  const konvaImageRef = useRef(null);
  const transformerRef = useRef(null);
  const stageRef = useRef(null);

  // Update transformer when selection changes
  useEffect(() => {
    if (isSelected && konvaImageRef.current && transformerRef.current) {
      transformerRef.current.nodes([konvaImageRef.current]);
      transformerRef.current.getLayer().batchDraw();
    } else if (transformerRef.current) {
      transformerRef.current.nodes([]);
      transformerRef.current.getLayer().batchDraw();
    }
  }, [isSelected, imageObj]);

  // helper: get displayed image data (from Konva image node)
  const getDisplayedImageData = () => {
    const node = konvaImageRef.current;
    if (!node || !node.image()) return null;
    const canvas = document.createElement("canvas");
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(node.image(), 0, 0, imageSize.width, imageSize.height);
    return ctx.getImageData(0, 0, imageSize.width, imageSize.height);
  };

  // internal: set Konva image from ImageData (does NOT change filteredImageData)
  const _setImageObjFromImageData = (imageData) => {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCanvas.getContext("2d").putImageData(imageData, 0, 0);
    const img = new window.Image();
    img.onload = () => setImageObj(img);
    img.src = tempCanvas.toDataURL();
  };

  // set displayed image; optionally also set filteredImageData (so filters update base)
  const setDisplayedFromImageData = (imageData, setAsFiltered = false) => {
    if (setAsFiltered) setFilteredImageData(imageData);
    _setImageObjFromImageData(imageData);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const img = new window.Image();
      img.onload = () => {
        const maxW = 800;
        const scale = Math.min(maxW / img.width, 1);
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);
        setImageSize({ width: w, height: h });
        setImageObj(img);
        setIsSelected(true);

        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, w, h);
        const imageData = ctx.getImageData(0, 0, w, h);
        setOriginalImageData(imageData);
        setFilteredImageData(imageData);
        console.log("imageData", imageData);
        // display as-is (no stroke)
        _setImageObjFromImageData(imageData);
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
  };

  // apply unsharp but operate on filteredImageData (base) and update base
  const applyUnsharp = (times = 2) => {
    if (!filteredImageData && !imageObj)
      return alert("Vui lòng tải ảnh trước.");
    let current =
      filteredImageData || getDisplayedImageData() || originalImageData;
    for (let t = 0; t < times; t++) {
      current = unsharpMaskImageData(current, amount, radius, threshold);
    }
    // update base (filtered) and then update displayed image (with stroke overlay if any)
    setFilteredImageData(current);
    if (strokeSize > 0) {
      const stroked = overlayStrokeOnImage(
        current,
        originalImageData || current,
        strokeSize,
        strokeColor
      );
      setDisplayedFromImageData(stroked, false);
    } else {
      setDisplayedFromImageData(current, false);
    }
  };

  const applyGray = () => {
    if (!filteredImageData && !imageObj)
      return alert("Vui lòng tải ảnh trước.");
    const base =
      filteredImageData || getDisplayedImageData() || originalImageData;
    const d = new Uint8ClampedArray(base.data);
    for (let i = 0; i < d.length; i += 4) {
      const gray = Math.round(
        0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
      );
      d[i] = d[i + 1] = d[i + 2] = gray;
    }
    const newImageData = new ImageData(d, base.width, base.height);
    setFilteredImageData(newImageData);
    if (strokeSize > 0) {
      const stroked = overlayStrokeOnImage(
        newImageData,
        originalImageData || newImageData,
        strokeSize,
        strokeColor
      );
      setDisplayedFromImageData(stroked, false);
    } else {
      setDisplayedFromImageData(newImageData, false);
    }
  };

  const applyThresholdSimple = () => {
    if (!filteredImageData && !imageObj)
      return alert("Vui lòng tải ảnh trước.");
    const base =
      filteredImageData || getDisplayedImageData() || originalImageData;
    const d = new Uint8ClampedArray(base.data);
    for (let i = 0; i < d.length; i += 4) {
      const gray = (d[i] + d[i + 1] + d[i + 2]) / 3;
      const v = gray > 127 ? 255 : 0;
      d[i] = d[i + 1] = d[i + 2] = v;
    }
    const newImageData = new ImageData(d, base.width, base.height);
    setFilteredImageData(newImageData);
    if (strokeSize > 0) {
      const stroked = overlayStrokeOnImage(
        newImageData,
        originalImageData || newImageData,
        strokeSize,
        strokeColor
      );
      setDisplayedFromImageData(stroked, false);
    } else {
      setDisplayedFromImageData(newImageData, false);
    }
  };

  const increaseContrast = (amt = 40) => {
    if (!filteredImageData && !imageObj)
      return alert("Vui lòng tải ảnh trước.");
    const base =
      filteredImageData || getDisplayedImageData() || originalImageData;
    const d = new Uint8ClampedArray(base.data);
    const factor = (259 * (amt + 255)) / (255 * (259 - amt));

    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.round(
        Math.min(255, Math.max(0, factor * (d[i] - 128) + 128))
      );
      d[i + 1] = Math.round(
        Math.min(255, Math.max(0, factor * (d[i + 1] - 128) + 128))
      );
      d[i + 2] = Math.round(
        Math.min(255, Math.max(0, factor * (d[i + 2] - 128) + 128))
      );
    }

    const newImageData = new ImageData(d, base.width, base.height);
    setFilteredImageData(newImageData);
    if (strokeSize > 0) {
      const stroked = overlayStrokeOnImage(
        newImageData,
        originalImageData || newImageData,
        strokeSize,
        strokeColor
      );
      setDisplayedFromImageData(stroked, false);
    } else {
      setDisplayedFromImageData(newImageData, false);
    }
  };

  const resetImage = () => {
    if (!originalImageData) return;
    setFilteredImageData(originalImageData);
    setDisplayedFromImageData(originalImageData, false);
    // Reset transform
    if (konvaImageRef.current) {
      konvaImageRef.current.x(0);
      konvaImageRef.current.y(0);
      konvaImageRef.current.scaleX(1);
      konvaImageRef.current.scaleY(1);
      konvaImageRef.current.rotation(0);
    }
  };

  // ---- Stroke handling (live) ----
  const applyStrokeLive = (size = strokeSize, colorHex = strokeColor) => {
    // operate on the filteredImageData (base). This preserves filters.
    const base =
      filteredImageData || getDisplayedImageData() || originalImageData;
    if (!base) return;

    const s = parseInt(size, 10) || 0;
    if (s === 0) {
      // show base (filters without stroke)
      setDisplayedFromImageData(base, false);
      return;
    }

    console.log("base", base);

    // mask source: prefer original alpha (keeps consistent silhouette), fallback to base
    const maskSource = originalImageData || base;
    const stroked = overlayStrokeOnImage(base, maskSource, s, colorHex);
    setDisplayedFromImageData(stroked, false);
  };

  const handleStrokeSizeChange = (e) => {
    const val = parseInt(e.target.value, 10) || 0;
    setStrokeSize(val);
    applyStrokeLive(val, strokeColor);
  };

  const handleStrokeColorChange = (e) => {
    const val = e.target.value;
    setStrokeColor(val);
    applyStrokeLive(strokeSize, val);
  };

  const handleStageClick = (e) => {
    // Clicked on stage - deselect if not clicking on the image
    if (e.target === e.target.getStage()) {
      setIsSelected(false);
    } else if (e.target === konvaImageRef.current) {
      setIsSelected(true);
    }
  };

  return (
    <div style={{ maxWidth: 980, margin: "20px auto", padding: 16 }}>
      <h1 style={{ fontSize: 18 }}>Unsharp Mask (React + react-konva)</h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginTop: 8,
        }}
      >
        <div>
          <label>Chọn ảnh (PNG có alpha được khuyến nghị)</label>
          <br />
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>

        <div>
          <label>Preview width limit (px)</label>
          <div>auto 800</div>
        </div>

        <div>
          <label>Amount: {amount}</label>
          <input
            type="range"
            min="0"
            max="10"
            step="0.1"
            value={amount}
            onChange={(e) => setAmount(parseFloat(e.target.value))}
          />
        </div>

        <div>
          <label>Radius (sigma): {radius}</label>
          <input
            type="range"
            min="0"
            max="50"
            step="1"
            value={radius}
            onChange={(e) => setRadius(parseInt(e.target.value, 10))}
          />
        </div>

        <div>
          <label>Threshold: {threshold}</label>
          <input
            type="range"
            min="0"
            max="255"
            step="1"
            value={threshold}
            onChange={(e) => setThreshold(parseInt(e.target.value, 10))}
          />
        </div>

        <div>
          <label>Stroke size: {strokeSize}px</label>
          <input
            type="range"
            min="0"
            max="40"
            step="1"
            value={strokeSize}
            onChange={handleStrokeSizeChange}
          />

          <label style={{ marginTop: 6 }}>Stroke color:</label>
          <input
            type="color"
            value={strokeColor}
            onChange={handleStrokeColorChange}
          />
        </div>
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={() => applyUnsharp(2)}>Apply Unsharp Mask (x2)</button>
        <button onClick={applyGray}>Apply Grayscale</button>
        <button onClick={applyThresholdSimple}>Apply Threshold</button>
        <button onClick={() => increaseContrast(40)}>Increase Contrast</button>
        <button onClick={resetImage}>Reset</button>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 14, marginBottom: 8, color: "#666" }}>
          📌 Click on the image to select it, then drag to move or use the
          corners to scale and rotate. Click outside the image to deselect.
        </div>
        <Stage
          ref={stageRef}
          width={stageSize.width}
          height={stageSize.height}
          style={{ border: "1px solid #ddd", background: "transparent" }}
          onClick={handleStageClick}
        >
          <Layer>
            <KonvaImage
              ref={konvaImageRef}
              image={imageObj}
              width={imageSize.width}
              height={imageSize.height}
              draggable={true}
              onClick={() => setIsSelected(true)}
            />
            {isSelected && (
              <Transformer
                ref={transformerRef}
                rotateEnabled={true}
                enabledAnchors={[
                  "top-left",
                  "top-center",
                  "top-right",
                  "middle-left",
                  "middle-right",
                  "bottom-left",
                  "bottom-center",
                  "bottom-right",
                ]}
                boundBoxFunc={(oldBox, newBox) => {
                  // Limit resize
                  if (newBox.width < 10 || newBox.height < 10) {
                    return oldBox;
                  }
                  return newBox;
                }}
              />
            )}
          </Layer>
        </Stage>
      </div>
    </div>
  );
}
