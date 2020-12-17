import torch
from csv import reader
from transformers import GPT2Tokenizer

# Select Device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def load_models(model_path):
    # Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load Model
    model = torch.load(model_path)
    model = model.to('cuda')

    return tokenizer, model


def generate(text, model, tokenizer, num=1, length=70):
    '''
    Generate suggestions based on a text: 
    '''
    encoded_prompt = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    suggested_sentences = []
    sample_outputs = model.generate(
                              input_ids=encoded_prompt,
                              do_sample=True,   
                              top_k=50, 
                              max_length = length,
                              top_p=0.95, 
                              num_return_sequences=num, 
                              pad_token_id = tokenizer.eos_token_id)
    for i, sample_output in enumerate(sample_outputs):
        sentence = tokenizer.decode(sample_output, skip_special_tokens=False).split("\n")[0]
        print(i+1,":", sentence)
        suggested_sentences.append(sentence)

    return suggested_sentences


if __name__ == "__main__":
    # Load Model
    model_path = "/content/drive/My Drive/550/output/gpt/model_6"
    input_path = "/content/drive/My Drive/550/data/reddit/summary_test.txt"
    file_path = "/content/drive/My Drive/550/output/processed/reddit/gpt_out.txt"
    # file_path = "/content/drive/My Drive/550/output/processed/reddit/hybrid_out.txt"
    # file_path = "/content/drive/My Drive/550/output/processed/daily/gpt_out.txt"
    # file_path = "/content/drive/My Drive/550/output/processed/daily/hybrid_out.txt"
    tokenizer, model = load_models(model_path)

    with open(input_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            text = row.lower().strip()
            wrt_obj = open(file_path, 'a')
            wrt_obj.write(''.join(generate(text, model, tokenizer, return_seq, length))+'\n')
        wrt_obj.close()
    read_obj.close()