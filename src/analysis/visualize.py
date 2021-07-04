from bertviz import model_view, head_view
from bertviz.neuron_view import show
from bertviz.transformers_neuron_view import RobertaModel as VizModel, RobertaTokenizer as VizTokenizer
from transformers import RobertaModel, RobertaTokenizer


class BertVisualizer:

    def __init__(self, sentence1: str, sentence2: str, display_mode: str = 'light'):
        self.display_mode = display_mode
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large', output_attentions=True)

        self.sentence1 = sentence1
        self.sentence2 = sentence2
        inputs = self.tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        self.attention = self.model(input_ids)[-1]
        input_id_list = input_ids[0].tolist()
        self.tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)

    def attention_view(self):
        model_view(self.attention, self.tokens, display_mode=self.display_mode)

    def head_view(self):
        head_view(self.attention, self.tokens)

    def neuron_view(self):
        model = VizModel.from_pretrained('roberta-large')
        tokenizer = VizTokenizer.from_pretrained('roberta-large')
        show(model, 'roberta', tokenizer, self.sentence1, self.sentence2, layer=2, head=0,
             display_mode=self.display_mode)
