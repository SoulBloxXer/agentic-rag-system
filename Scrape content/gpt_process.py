from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import json
import openai
import os

def setup_openai_client():
    # Initialize the client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
    )
    return client

def chat_with_gpt4(client, messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    # Initialize the client
    client = setup_openai_client()
    
    # Read your text file
    with open('extracted_articles.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize conversation with the text
    messages = [
        {"role": "system", "content": "You are a helpful assistant that processes and analyzes text. You will receive a large text file and can process it in batches of as much as possible, whilst maintaining context throughout the conversation."},
        {"role": "user", "content": f"""I'm about to give you a text file which i want you to process. It's quite long so you'll need to process it in chunks. Do as much as you can at a time.
         I'd like you to remember this throughout our whole conversation.
         ..So then, here's your task:
         The text file is essentially the entire knowledge base of a website, and i intend to format it as best as possible for an llm to understand. Each section in the file is a different article which is found in a distinct webpage; the url is given at the start of each section.
         I'd like the different sections .to be retained, as well as the url of where it's found. You must keep all the text the same (whilst fixing any grammar issues). Instead, I just want you to better format the text (ideally in markdown) so that it's optimised for an llm to use it for rag.
         
         ---

         Here is a small sample of the text file content:
         "URL: http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-celtic-fish-game

        # Case Study - Celtic Fish & Game

        Introduction:

        We were contacted by Celtic Fish and Game, the Cornish company who were established in 1985, quickly became a family run business based in St. Ives. They supply the catering trade and the company consists of Dad Geoff, mum Sylvia, daughters Emma, Naomi and Hannah and niece Zoe who, all play vital roles in the day to day running of Celtic Fish and Game. Geoff, a fisherman with over forty years experience in the catching sector brings a wealth of knowledge about all aspects of the fishing industry.

        Celtic Fish and Game began sourcing fresh fish from local markets and boats and wild seasonal game from local estates. In its early days the product range spanned a single A4 page listing 35 local fish species and whichever game was in season at the time. Their Mantra "As far as possible our products are sourced locally".

        Celtic Fish and Game remarked:

        "We buy and process feathered and ground game from hunters throughout Cornwall and Devon. Game meat is a product area which is becoming more and more popular and is one of the few product ranges along with fish that can be guaranteed wild and traceable. Another factor responsible in increasing the popularity of game is the consistency it provides in aspects of quality, price and flavour.

        In addition to the Fish and Game we stock over one thousand fine food items. Our list has evolved from demand by chefs. The chef decides the ingredients they wish to work with and we do our upmost to source these products and so the list grows.

        Sourcing as locally as possible is very important to us. In addition to the fish and game there is a wealth of local produce available. We buy local shellfish, meat, cheese and chutneys.

        We have a smokehouse on-site where old techniques are used to produce exciting new products with local ingredients. We also smoke on request, so if you have an idea just let us know.

        As a company we have evolved our range by customer demand, therefore our products always reflect current trends. Continue to tell us want you want and we will always do our best to supply it."


        ================================================================================


        URL: http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-hogsbottom-garden-delights

        # Hogsbottom Garden Delights

        Introduction:

        We were contacted by an award winning artisan Jam, chutney and preserve producer that currently handmake in small batches jams chutneys and dressings to maximise the flavour and they use local fresh produce whenever seasonally possible.

        With a forty foot long row of raspberry canes to use each year at the BOTTOM of the GARDEN we eventually got fed up with making jam and liqueurs.

        We developed and bottled some salad dressings and fruit vinegars to give to friends and family as presents. This proved a bit of a hit so more flavours were made!

        At the same time we released hedgehogs into the Bottom of our garden and the name was born!

        Hogsbottom, are passionate and only natural ingredients of the finest quality and absolutely NO artificial colours, flavours or preservatives go into their products. Plus all of their products are also Gluten FREE! Except for our real ale products.

        Over 20 Awards across the range of products over the last few years, and to continue this tradition they needed a wrapping solution that showcased their fine product. We were drafted in to provide an l sealer and tunnel to wrap trays of 3 6 and 12 bottles/jars in a presentation/travel format.

        This product requirement as with many of our solutions is in our DNA so it wasn't long before we had their product in our machines via a demo.

        Needless to say Hogsbottom were impressed, and naturally went ahead with their purchase. We provided a solution that would be easy to use from an end user perspective, was efficient, eliminated waste, and provided an environmentally friendly presentation product. Polyolefin Shrink Film was used in an L sealer and tunnel line pefect for increased output should it be required, but also for variable sizes of trays etc.

        Links here: …..

        Company Budget: £6,000

        Purchase Costs: £4,000

        Production Upturn: 685%

        Packaging costs reduced by 90%

        Project Comment:

        Shrink Wrapping Supplies, supplied a first class service from start to finish, and allowed our end users an easy to use solution, a value added product to our already excellently presented Jars/Bottles.

        ================================================================================
        "
         
         ---

         Before you start, does everything i said make sense?"""}
    ]
    
    # Start the conversation loop
    while True:
        # Get response from GPT
        response = chat_with_gpt4(client, messages)
        if response:
            print("\nGPT-4:", response)
            
            # Add the response to the conversation history
            messages.append({"role": "assistant", "content": response})
            
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            # Add user input to the conversation history
            messages.append({"role": "user", "content": user_input})
        else:
            print("Error getting response from GPT-4")
            break

if __name__ == "__main__":
    main() 