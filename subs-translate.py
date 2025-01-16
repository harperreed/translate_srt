import srt
import argparse
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()
from tqdm import tqdm

# Set your OpenAI API key


def read_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return list(srt.parse(file))

def write_srt(file_path, subtitles):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(srt.compose(subtitles))

def translate_text(text):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional translator. Translate the following Japanese text to English. Maintain the original meaning and nuance as much as possible."},
        {"role": "user", "content": text}
    ])

    return response.choices[0].message.content.strip()

def translate_srt(input_file, output_file):
    subtitles = read_srt(input_file)
    
    translated_subtitles = []
    
    for sub in tqdm(subtitles, desc="Translating subtitles"):
        translated_content = translate_text(sub.content)
        
        translated_sub = srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=translated_content
        )
        
        translated_subtitles.append(translated_sub)
    
    write_srt(output_file, translated_subtitles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate SRT subtitles from Japanese to English')
    parser.add_argument('input_file', help='Input SRT file path')
    parser.add_argument('output_file', help='Output SRT file path')
    
    args = parser.parse_args()
    
    translate_srt(args.input_file, args.output_file)
    print(f"Translation completed. Output file: {args.output_file}")
