import asyncio
from firecrawl import AsyncFirecrawlApp
from pydantic import BaseModel, Field
from typing import Any, Optional, List
import json
import asyncio.exceptions

class ExtractSchema(BaseModel):
    article_md: str  # Now just a single string instead of an array

async def process_single_url(app, url, timeout=300):  # Added timeout parameter
    try:
        # Add timeout to prevent hanging
        async with asyncio.timeout(timeout):
            response = await app.extract(
                urls=[url],  # Process one URL at a time
                prompt='Extract everything from the main article i.e. all the way from heading 1, up till before heading 4 "Did you find this article useful?". Extract it in clean markdown format, and make sure to add new lines where appropriate.',
                schema=ExtractSchema.model_json_schema()
            )
            return response
    except asyncio.exceptions.TimeoutError:
        print(f"Timeout processing {url} after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

async def main():
    app = AsyncFirecrawlApp(api_key='fc-5a9e6fbdd5b7453db3bd8ecda18f3967')
    
    # List of URLs to process
    urls = [
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-celtic-fish-game",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-hogsbottom-garden-delights",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/buffets-4-business",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/impressed-laundry",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/almighty-foods",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/cosmetics",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/dairies",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/dvd-s",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/cd-s",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/dietary-supplements",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/laboratories",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/promotional-packer",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/distribution-company",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/printers",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/label-manufacturer-intro",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/mailing-company",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/micro-brewery",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/hampers",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/kitchen-cabinet-manufacturer",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/cash-carry-warehouse",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/exam-papers",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/confectioner",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-splitting-seals",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/dog-ears-triangles",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/problems-with-sealing",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-splitting-seals",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-hot-spots-or-holes",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-ballooning",
            "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-fish-eyes",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/all-in-one-chamber-machine-5540",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-l-sealer-tunnel-separate-units",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-l-sealers-automation-range-high-speed-lines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-automatic-side-sealing-machine-overview",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-automatic-side-sealing-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-semi-automatic-sleeve-sealer",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-manual-sleeve-sealer-shrink-tunnel",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-front-loading-sleeve-sealer",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-automatic-side-sealing-machine-2",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-l-sealer-tunnel-combined",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-large-l-sealer-and-tunnel-fl7555",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-large-semi-automatic-l-sealer-tunnel-sm604",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-sm-8040-shrink-tunnel",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-heat-shrink-tunnel-for-food-and-non-food-applications",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-heat-shrink-tunnel-general-overview",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-labeling-shrink-tunnel",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-shrink-sleeve-wrapper-plain-or-printed",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-steam-label-sleeve-shrink-tunnel-and-steam-generator",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/a-semi-automatic-turntable-stretch-wrapper-saves-one-year-s-salary",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/which-machine-pallet-wrapper-should-you-choose",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/hand-pallet-stretch-wrap-for-picking-orders",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machine-pallet-wrap-vs-hand-held-pallet-wrap",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/spiral-wrapper-machines-for-faster-production-lines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-pallet-stretch-wrap-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-handheld-strapping-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/benefits-of-a-handheld-strapping-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/knowledge-base-article-semi-automatic-strapping-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/knowledge-base-article-automatic-banding-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/non-powered-conveyor",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/semi-circle-return-powered-conveyor-belt",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-automatic-vertical-stretch-wrapper-machines-save",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/auto-sealing-machine-for-juice-packaging",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/buying-auto-packaging-machines-for-rice-mills",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/shrink-film-shelf-life",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/replace-hand-wrap-with-a-stretch-wrapper-for-pallet-wrapping",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/do-you-need-a-cheap-machine-pallet-wrapper",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/help-with-buying-a-pallet-wrapper",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/what-is-machine-sleeve-wrapping",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/why-or-how-should-i-shrink-wrap-fruit-and-veg",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/what-is-the-cheapest-supplier-for-shrink-film",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/compare-cheap-shrink-wrapping-machinery-online",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/can-i-save-money-on-thick-shrink-wrap-film",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/top-10-reasons-to-shrink-wrap-your-products",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/why-a-automatic-horizontal-stretch-wrapper-for-furniture",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/choosing-the-best-shrink-wrap-machine-for-warehouses",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/choosing-the-best-shrink-wrap-machine-for-retailers",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/manual-shrink-packaging-machine-vs-second-hand-auto",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/upgrading-from-manual-shrink-wrap-equipment",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/automated-vs-manual-shrink-wrap-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/semi-automatic-shrink-wrap-machine-for-print",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/shrink-wrap-machinery-for-food",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/secondhand-packaging-machinery",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/a-sealing-machine-for-food-packaging",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/heavy-duty-shrink-wrap-film-v-heavy-duty-shrink-film",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/custom-built-sealing-machines-for-trays",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/shrink-wrap-machinery-and-new-start-ups",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/planning-for-high-speed-shrink-wrapping-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/used-and-secondhand-shrink-wrap-machine-for-budgeting",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-to-choose-the-correct-shrink-wrap-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/the-world-of-shrink-wrap-machines-and-product-possibilities",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-we-can-supply-customised-shrink-wrap-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/can-i-hire-shrink-wrapping-equipment-prior-to-purchasing",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/do-i-need-shrink-wrap-machine-maintenance-and-servicing",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/cost-savings-associated-with-a-part-ex-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/exploring-belt-driven-box-sealer-machines-an-in-depth-guide",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-automatically-adjusting-belt-driven-box-taping-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-linear-friction-feeder-machines-a-comprehensive-overview",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-finger-push-infeeder-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-friction-feeders",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-inkjet-coder-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/the-top-and-bottom-labeller-machine-an-essential-tool-for-efficient-packaging",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-inkjet-and-labelling-systems-a-comprehensive-guide",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-self-propelled-robot-stretch-wrapping-machines",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/special-robotics-for-self-wrapping-machines-an-in-depth-overview",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/we-show-you-that-polyolefin-is-recyclable",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/are-there-any-truly-biodegradable-shrink-films",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/saving-the-planet",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/are-there-any-truly-biodegradable-shrink-films-2",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-to-work-out-shrink-film-sizes",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-to-use-shrink-film-for-added-security",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/using-shrink-films-for-marketing-purposes",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/protection-for-your-products-from-shrink-film",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/why-choose-shrink-wrapping",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/why-choose-a-shrink-wrap-chamber-hood-machine",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/difference-between-types-of-shrink-wrap",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/risk-assessing",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/quality-control-in-a-semi-automatic-shrink-wrap-machine-world",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/what-are-our-service-contract-visits",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/can-i-trial-a-free-shrinkwrap-machine-demo",
            "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/pdf-manuals-machine-operating-instructions"
        ]

    # Process each URL and save results incrementally
    processed_urls = set()
    
    # Try to load already processed URLs from a progress file
    try:
        with open('progress.txt', 'r', encoding='utf-8') as f:
            processed_urls = set(line.strip() for line in f)
    except FileNotFoundError:
        pass

    # Open the output file in append mode
    with open('extracted_articles.txt', 'a', encoding='utf-8') as f:
        for url in urls:
            if url in processed_urls:
                print(f"Skipping already processed URL: {url}")
                continue
                
            print(f"Processing: {url}")
            response = await process_single_url(app, url)
            
            if response and response.data and response.data.get('article_md'):
                content = response.data['article_md']
                if content:
                    # Write immediately after successful processing
                    f.write(f"URL: {url}\n\n{content}\n\n{'='*80}\n\n")
                    f.flush()  # Ensure it's written to disk
                    
                    # Save progress
                    with open('progress.txt', 'a', encoding='utf-8') as progress_file:
                        progress_file.write(f"{url}\n")
                        progress_file.flush()
                    
                    processed_urls.add(url)
            
            # Add a small delay between requests
            await asyncio.sleep(1)

    print(f"Processing complete. Results saved to 'extracted_articles.txt'")

asyncio.run(main())