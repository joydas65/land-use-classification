# TerraClass web interface

This Tailwind CSS/Next.js browser interface turns the versioned TerraClass inference contract into
an interactive portfolio demonstration. It checks model readiness, validates image
uploads, requests ranked predictions, and displays confidence, dimensions,
latency, model version, and request provenance.

## Local development

Use Node.js 24. Start the FastAPI service from the repository root,
then start the interface from this directory:

```bash
pnpm install
pnpm run dev
```

The default API address is `http://localhost:8000`. Set
`NEXT_PUBLIC_TERRACLASS_API_URL` at build time when the API is hosted elsewhere.

## Validation

```bash
pnpm run lint
pnpm test
pnpm run build:vercel
```

The test command builds the production worker and verifies its server-rendered
content, accessibility contracts, model scope language, and versioned API paths.
The Vercel command separately verifies the native Next.js production artifact.

## Deployment

The public frontend is available at
[terraclass-land-use-classification.vercel.app](https://terraclass-land-use-classification.vercel.app).
`vercel.json` pins the native Next.js build and reproducible `npm ci` install. The deployed UI
truthfully reports the model as offline until `NEXT_PUBLIC_TERRACLASS_API_URL` points to a hosted
HTTPS FastAPI origin.
