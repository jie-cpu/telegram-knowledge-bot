
# 
## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Run the app:
   `npm run dev`

## Deploy To Cloudflare Pages (No API Key Exposure)

This app uses a Cloudflare Pages Function at `/api/chat` so your Gemini key stays server-side.

1. Build the app:
   `npm run build`
2. Create a Cloudflare Pages project (build output directory: `dist`)
3. Add secret in Pages project settings:
   `GEMINI_API_KEY=<your-key>`
4. Deploy with Wrangler (optional CLI flow):
   `npx wrangler pages deploy dist --project-name <your-project-name>`

For local Cloudflare runtime testing:

`npx wrangler pages dev dist`
