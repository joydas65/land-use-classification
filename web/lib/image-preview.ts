import { decode, type TiffIfd } from "tiff";

const TIFF_MIME_TYPE = "image/tiff";
const MAX_INPUT_PIXELS = 25_000_000;
const MAX_PREVIEW_PIXELS = 4_000_000;

function byteValue(value: number, maximum: number) {
  if (!Number.isFinite(value) || maximum <= 0) return 0;
  return Math.max(0, Math.min(255, Math.round((value / maximum) * 255)));
}

function previewDimensions(width: number, height: number) {
  const scale = Math.min(1, Math.sqrt(MAX_PREVIEW_PIXELS / (width * height)));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export function tiffToPreviewPixels(image: TiffIfd) {
  const { width, height, components, data, maxSampleValue, type } = image;
  const inputPixels = width * height;

  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || inputPixels <= 0) {
    throw new Error("The TIFF has invalid dimensions.");
  }
  if (inputPixels > MAX_INPUT_PIXELS) {
    throw new Error("The TIFF exceeds the 25-megapixel preview limit.");
  }
  if (components !== 1 && components !== 3 && components !== 4) {
    throw new Error(`TIFF previews do not support ${components} color components.`);
  }

  const outputSize = previewDimensions(width, height);
  const rgba = new Uint8ClampedArray(outputSize.width * outputSize.height * 4);

  for (let outputY = 0; outputY < outputSize.height; outputY += 1) {
    const sourceY = Math.min(height - 1, Math.floor((outputY * height) / outputSize.height));
    for (let outputX = 0; outputX < outputSize.width; outputX += 1) {
      const sourceX = Math.min(width - 1, Math.floor((outputX * width) / outputSize.width));
      const sourceOffset = (sourceY * width + sourceX) * components;
      const outputOffset = (outputY * outputSize.width + outputX) * 4;

      if (components === 1) {
        const sample = byteValue(data[sourceOffset], maxSampleValue);
        const grayscale = type === 0 ? 255 - sample : sample;
        rgba[outputOffset] = grayscale;
        rgba[outputOffset + 1] = grayscale;
        rgba[outputOffset + 2] = grayscale;
        rgba[outputOffset + 3] = 255;
      } else {
        rgba[outputOffset] = byteValue(data[sourceOffset], maxSampleValue);
        rgba[outputOffset + 1] = byteValue(data[sourceOffset + 1], maxSampleValue);
        rgba[outputOffset + 2] = byteValue(data[sourceOffset + 2], maxSampleValue);
        rgba[outputOffset + 3] = components === 4
          ? byteValue(data[sourceOffset + 3], maxSampleValue)
          : 255;
      }
    }
  }

  return { data: rgba, width: outputSize.width, height: outputSize.height };
}

function canvasToPngBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("The browser could not create a PNG preview."));
      }
    }, "image/png");
  });
}

export async function createImagePreviewUrl(file: File) {
  if (file.type !== TIFF_MIME_TYPE) {
    return URL.createObjectURL(file);
  }

  const [image] = decode(new Uint8Array(await file.arrayBuffer()), { pages: [0] });
  if (!image) {
    throw new Error("The TIFF does not contain a readable image.");
  }

  const preview = tiffToPreviewPixels(image);
  const imageData = new ImageData(preview.data, preview.width, preview.height);
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("The browser could not prepare the TIFF preview.");
  }
  context.putImageData(imageData, 0, 0);

  return URL.createObjectURL(await canvasToPngBlob(canvas));
}
