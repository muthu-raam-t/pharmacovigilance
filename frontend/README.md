# Pharmacovigilance Advisory — Frontend

Single-page React application for the drug–disease advisory workflow. It calls the FastAPI backend's `POST /analyze` endpoint and shows:

- the suitability verdict with its evidence source (clinical reference, human annotation or model prediction with confidence)
- **Why this verdict**: the factor-level score breakdown
- the ranked side-effect profile (frequency bucket, SIDER / OnSIDES source, FDA boxed warnings)
- suggested alternatives with evidence tier and side-effect burden
- supporting literature sentences with the drug and disease highlighted
- the plain-language explanation

## Run

```bash
npm install
npm start          # http://localhost:3000
```

The API address defaults to `http://localhost:8000`. To change it, copy `.env.example` to `.env.local` and set `REACT_APP_API_URL`.

## Test and build

```bash
CI=true npm test   # component tests (API mocked)
npm run build      # production build in build/
```
