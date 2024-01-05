# Databricks notebook source
# MAGIC %md
# MAGIC # Google Cloud setup
# MAGIC This lab outlines the steps you will need to take to use Google Cloud and Vertex AI for your own projects.
# MAGIC
# MAGIC Note: To try this out yourself, please download this notebook and run it on your local computer.
# MAGIC
# MAGIC Create a Google Cloud Project
# MAGIC Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services including managing APIs, enabling billing, adding and removing collaborators, and managing permissions for Google Cloud resources.
# MAGIC
# MAGIC Your usage of Google Cloud tools is always associated with a project.
# MAGIC
# MAGIC You will be prompted to create a new project the first time you visit the Cloud Console
# MAGIC
# MAGIC Note that you can create a free project which includes a 90-day $300 Free Trial.
# MAGIC
# MAGIC Learn more about projects here.
# MAGIC
# MAGIC Set up Billing
# MAGIC A Cloud Billing account is used to define who pays for a given set of resources, and it can be linked to one or more projects. Project usage is charged to the linked Cloud Billing account.
# MAGIC
# MAGIC Within your project, you can configure billing by selecting "Billing" in the menu on the left.
# MAGIC
# MAGIC select billing
# MAGIC
# MAGIC Make sure that billing is enabled for your Google Cloud project, click here to learn how to confirm that billing is enabled.
# MAGIC
# MAGIC Enable APIs
# MAGIC Once you have a project set up with a billing account, you will need to enable any services you want to use.
# MAGIC
# MAGIC Click here to enable the following APIs in your Google Cloud project:
# MAGIC
# MAGIC Vertex AI
# MAGIC BigQuery
# MAGIC IAM
# MAGIC Create service account
# MAGIC A service account is a special kind of account typically used by an application or compute workload, such as a Compute Engine instance, rather than a person. A service account is identified by its email address, which is unique to the account. To learn more, check out this intro video.
# MAGIC
# MAGIC You will need to create a service account and give it access to the Google Cloud services you want to use.
# MAGIC
# MAGIC 1. Go to the Create Service Account page and select your project
# MAGIC 2. Give the account a name (you can pick anything)
# MAGIC create service account
# MAGIC
# MAGIC Grant the account the following permissions
# MAGIC grant permissions
# MAGIC
# MAGIC Create Service Account key
# MAGIC Once you have created your service account, you need to create a key.
# MAGIC
# MAGIC 1. Select your newly created service account then click. ADD KEY -> create new key.
# MAGIC 2. Select JSON key type and click create
# MAGIC JSON key
