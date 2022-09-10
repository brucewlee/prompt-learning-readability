import torch

class HF_Tester:
    def __init__ (
        self,
        model: object,
        eval_dataloader: object,
        device: object,
        ) -> None:

        self.model = model
        self.eval_dataloader = eval_dataloader
        self.device = device
    
    def test(self) -> list:
        all_pred_labels = []
        self.model.eval()
        for idx, batch in enumerate(self.eval_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
            
            pred_labels = logits.argmax(axis = -1).flatten().tolist()
            all_pred_labels.extend(pred_labels)
        
        return all_pred_labels

if __name__ == "__main__":
    from dataset import HF_DatasetConstructor
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = 'roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = 'ckpt/roberta-base-2e-05-8-13epochs', num_labels = 2)

    TestDC = HF_DatasetConstructor(
        data_path = 'data/test_subtask1_text.json',
        file_format = 'ndjson',
        tokenizer = tokenizer,
        text_field = 'text',
        max_seq_length = 256
    )
    print(TestDC[0])
    sample_test_dataloader = DataLoader(TestDC, batch_size=8, shuffle=False)
    for batch in sample_test_dataloader:
        print(batch)
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    HFT = HF_Tester(model = model, eval_dataloader = sample_test_dataloader, device = device)
    pred_labels = HFT.test()

    """CodaLab Format"""
    print("CodaLab Format")
    import utils.file_handler as f_handler
    import numpy as np

    df = f_handler.load_ndjson_and_return_pandas('data/test_subtask1_text.json')
    df['index'] = np.arange(len(df))
    df["prediction"] = pred_labels
    df_coda = df[["index","prediction"]]
    print(df.head())
    print(df_coda.head())
    f_handler.get_pandas_and_save_ndjson(df, 'data/roberta_base_13_2e-5_8_pred_test_subtask1_text.json')
    f_handler.get_pandas_and_save_ndjson(df_coda, 'data/coda/roberta_base_13_2e-5_8_coda_test_subtask1_text.json')
        
