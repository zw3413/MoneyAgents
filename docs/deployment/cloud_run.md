# Deploying to Google Cloud Run

This guide explains how to deploy the Trading Assistant to Google Cloud Run.

## Prerequisites

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Enable required APIs in your GCP project:
   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     containerregistry.googleapis.com
   ```
3. Authenticate with Google Cloud and grant docker permit to operate with GCP repository:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker us-central1-docker.pkg.dev
   ```

## Deployment Steps

1. Set up environment variables:
   ```bash
   export PROJECT_ID=$(gcloud config get-value project)
   export REGION=us-central1  # or your preferred region
   ```

2. Build and deploy using Cloud Build:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

3. enable the api privilage for all user
   ```bash
   gcloud run services add-iam-policy-binding trading-asst --region=us-central1 --member="allUsers" --role="roles/run.invoker"
   ```

## Manual Deployment (Alternative)
1. Build the Docker image:
   ```bash
   docker build -t gcr.io/$PROJECT_ID/trading-asst .
   ```

2. Push to Container Registry:
   ```bash
   docker push gcr.io/$PROJECT_ID/trading-asst
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy trading-asst \
     --image gcr.io/$PROJECT_ID/trading-asst/trading-asst \
     --platform managed \
     --region $REGION \
     --allow-unauthenticated \
   ```


## Environment Variables

The following environment variables are required to set if using manually deployment:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PORT`: Set automatically by Cloud Run (default: 8000)
- `HOST`: Set automatically by Cloud Run (default: 0.0.0.0)

Optional environment variables:
- `DEBUG`: Set to "false" in production
- `ENVIRONMENT`: Set to "production" in production

## Monitoring and Logging

1. View logs:
   ```bash
   gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=trading-asst"
   ```

2. View service metrics:
   ```bash
   gcloud run services describe trading-asst
   ```

## Security Considerations

1. Store sensitive environment variables (like API keys) in Secret Manager:
   ```bash
   # Create secret
   echo -n "your-api-key" | gcloud secrets create openai-api-key --data-file=-
   
   # Grant access to Cloud Run
   gcloud secrets add-iam-policy-binding openai-api-key \
     --member="serviceAccount:$PROJECT_ID-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

2. Update the deployment to use secrets:
   ```bash
   gcloud run deploy trading-asst \
     --image gcr.io/$PROJECT_ID/trading-asst \
     --set-secrets OPENAI_API_KEY=openai-api-key:latest
   ```

## Troubleshooting

1. If the service fails to start, check the logs:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=trading-asst"
   ```

2. To test the container locally:
   ```bash
   docker run -p 8080:8080 \
     -e OPENAI_API_KEY=your-api-key \
     gcr.io/$PROJECT_ID/trading-asst
   ```

3. Common issues:
   - Memory limits: Increase memory allocation in Cloud Run if needed
   - Cold starts: Consider using minimum instances
   - Timeouts: Adjust timeout settings for long-running operations 