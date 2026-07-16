import assert from "node:assert/strict";
import test from "node:test";

import { tiffToPreviewPixels } from "../lib/image-preview.ts";

test("converts eight-bit RGB TIFF pixels to opaque RGBA preview pixels", () => {
  const preview = tiffToPreviewPixels({
    width: 2,
    height: 1,
    components: 3,
    data: new Uint8Array([255, 0, 0, 0, 128, 255]),
    maxSampleValue: 255,
    type: 2,
  });

  assert.deepEqual(
    { width: preview.width, height: preview.height, data: [...preview.data] },
    {
      width: 2,
      height: 1,
      data: [255, 0, 0, 255, 0, 128, 255, 255],
    },
  );
});

test("honors the TIFF white-is-zero grayscale interpretation", () => {
  const preview = tiffToPreviewPixels({
    width: 2,
    height: 1,
    components: 1,
    data: new Uint8Array([0, 255]),
    maxSampleValue: 255,
    type: 0,
  });

  assert.deepEqual([...preview.data], [255, 255, 255, 255, 0, 0, 0, 255]);
});

test("rejects TIFF previews beyond the API pixel limit", () => {
  assert.throws(
    () => tiffToPreviewPixels({
      width: 5001,
      height: 5000,
      components: 3,
      data: new Uint8Array(),
      maxSampleValue: 255,
      type: 2,
    }),
    /25-megapixel preview limit/,
  );
});
