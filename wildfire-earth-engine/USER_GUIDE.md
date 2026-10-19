# Wildfire Model Lab — Simple User Guide

The Wildfire Model Lab maps burned areas using live Sentinel-2 satellite imagery. The imagery is retrieved from Google Earth Engine, and the selected model runs locally in your web browser.

## Before you begin

- Preferably use a laptop or desktop computer with a GPU and a recent version of Chrome or Edge.
- If WebGPU is available, the model will use your GPU. Otherwise, it will automatically run on the CPU using WebAssembly, which may be slower.
- You need a Google account with access to Google Earth Engine.

## One-time Google setup

1. Create or select a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Earth Engine API** and register the project for Earth Engine access.
3. Open **APIs & Services → Credentials**.
4. Create an **OAuth 2.0 Client ID** with the application type **Web application**.
5. Add the following under **Authorized JavaScript origins**:

   ```text
   https://alishibli97.github.io
   ```

6. Copy the OAuth **Client ID**, which ends in `.apps.googleusercontent.com`.

> Only use the public Client ID. Never enter or share the OAuth client secret or a service-account key.

If the OAuth consent screen is in testing mode, add your Google account under **Audience → Test users**.

## Run the application

1. Open the [Wildfire Model Lab](https://alishibli97.github.io/wildfire-earth-engine/).
2. Use the selector at the top of the page:
   - **United States** — US wildfire-progression model.
   - **Canada** — Canadian FlameNet-S2 model.
3. Enter your Google Cloud **Project ID** and OAuth **Client ID**.
4. Select **Connect Google account** and complete Google authorization.
5. Select a location by clicking the map or entering its latitude and longitude.
6. Choose the area size using the **ROI width** or **Window** control.
7. Confirm the dates, or keep the dates provided by one of the example events.
8. Select **Find imagery and run model**.

The application will find suitable Sentinel-2 imagery through Earth Engine and display the burned-area prediction when processing finishes. Large areas require more memory and take longer, particularly when the model runs on the CPU.

## Important

This is a research demonstration, not an operational emergency or fire-warning system. Cloud, smoke, snow, unavailable imagery, and locations outside the model's training domain can affect the results.
