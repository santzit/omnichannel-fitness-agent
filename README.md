# Omnichannel Fitness Agent

This repository contains the MVP scaffolding for an omnichannel fitness agent application using FastAPI, Langgraph, PostgreSQL, and pgvector. 

## Setup Instructions

1. Install dependencies.
2. Set up the database.
3. Run the application.

### Endpoints

- `/health`: For health check.
- `/webhooks/meta`: Web hook for meta Whatsapp.

## Getting Started

Follow the instructions in this document to get up and running with the application.


## TODO
Use queue processing
`Webhook → Redis Queue → Worker → Agent`

Prevents WhatsApp timeout.
