#!/usr/bin/env python
"""
Orchestration script for comprehensive documentation scraping.
Runs all three scrapers and builds unified indexes.
"""

import argparse
import schedule
import time
from datetime import datetime
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

from scrapers.python_sdk_scraper import PythonSDKScraper
from scrapers.community_github_scraper import CommunityGitHubScraper
from scrapers.rest_api_scraper import RestAPIScraper
from build_unified_index import UnifiedSearchIndexBuilder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_scraping_pipeline(initial_setup: bool = False):
    """
    Run the complete scraping and indexing pipeline.
    
    Args:
        initial_setup: If True, performs comprehensive initial scraping
    """
    start_time = datetime.now()
    logger.info(f"Starting comprehensive scrape at {start_time}")
    
    try:
        # 1. Scrape Python SDK
        logger.info("Scraping Python SDK documentation...")
        sdk_scraper = PythonSDKScraper()
        python_sdk_data = sdk_scraper.scrape_full_documentation()
        
        # 2. Scrape GitHub Community
        logger.info("Scraping GitHub Community repositories...")
        github_scraper = CommunityGitHubScraper(
            github_token=os.getenv('GITHUB_TOKEN')
        )
        github_data = github_scraper.scrape_all_repositories()
        
        # 3. Scrape REST API
        logger.info("Scraping REST API documentation...")
        rest_scraper = RestAPIScraper()
        rest_api_data = rest_scraper.scrape_rest_api_docs()
        
        # 4. Build unified indexes
        logger.info("Building unified search indexes...")
        index_builder = UnifiedSearchIndexBuilder()
        index_stats = index_builder.build_comprehensive_index(
            python_sdk_data,
            github_data,
            rest_api_data
        )
        
        # 5. Generate llms.txt
        generate_llms_txt(python_sdk_data, github_data, rest_api_data)
        
        elapsed = datetime.now() - start_time
        logger.info(f"Scraping complete in {elapsed}. Stats: {index_stats}")
        
        return index_stats
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def generate_llms_txt(python_sdk_data, github_data, rest_api_data):
    """Generate optimized llms.txt file."""
    # Implementation...
    pass

def main():
    parser = argparse.ArgumentParser(description="DataRobot Documentation Scraper")
    parser.add_argument(
        "--initial-setup",
        action="store_true",
        help="Perform initial comprehensive scraping"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on schedule (daily at 2 AM)"
    )
    
    args = parser.parse_args()
    
    if args.initial_setup or not args.schedule:
        # Run once immediately
        run_complete_scraping_pipeline(initial_setup=args.initial_setup)
    
    if args.schedule:
        # Schedule daily updates
        schedule_time = os.getenv("SCRAPE_SCHEDULE", "02:00")
        schedule.every().day.at(schedule_time).do(run_complete_scraping_pipeline)
        
        logger.info(f"Scraping scheduler started (daily at {schedule_time}). Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    main()