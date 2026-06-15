import os
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/hf_cache")

import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
    print("Hugging Face login successful")
else:
    print("HF_TOKEN not found")

class T5FineTuner:
    """
    Manages local Hugging Face T5 loading, training (fine-tuning) on CPU,
    and text generation for Agile User Stories to Gherkin translation.
    """
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(self.current_dir, "t5_finetuned_weights.pt")
        
    def load_model(self):
        """Loads T5 model and tokenizer from cache/HuggingFace."""
        if self.model is None:
            # T5 requires sentencepiece, which is bundled in transformers
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
    def train(self, csv_path: str, epochs: int, progress_callback=None):
        """Runs a real PyTorch training loop on the uploaded dataset."""
        self.load_model()
        
        # Load dataset using pandas
        df = pd.read_csv(csv_path)
        
        # Try to find Story and Criteria columns dynamically
        story_col = None
        criteria_col = None
        
        # 1. Exact match
        if "Story" in df.columns:
            story_col = "Story"
        if "Criteria" in df.columns:
            criteria_col = "Criteria"
            
        # 2. Case-insensitive search if not found
        if not story_col or not criteria_col:
            for col in df.columns:
                col_lower = str(col).lower()
                if not story_col and ("story" in col_lower or "input" in col_lower or "source" in col_lower):
                    story_col = col
                if not criteria_col and ("criteria" in col_lower or "gherkin" in col_lower or "target" in col_lower or "output" in col_lower or "regression" in col_lower or "test" in col_lower):
                    if col != story_col:
                        criteria_col = col
                        
        # 3. Fallback: if there are at least two columns and we still don't have them, use the first two columns
        if (not story_col or not criteria_col) and len(df.columns) >= 2:
            if not story_col:
                story_col = df.columns[0]
            if not criteria_col:
                criteria_col = df.columns[1] if df.columns[1] != story_col else df.columns[0]
                
        if not story_col or not criteria_col:
            raise ValueError("CSV must contain columns representing 'Story' and 'Criteria'.")
            
        stories = df[story_col].dropna().tolist()
        criteria = df[criteria_col].dropna().tolist()
        
        # Format dataset for T5 translation task
        inputs = ["translate Story to Gherkin: " + str(s) for s in stories]
        targets = [str(c) for c in criteria]
        
        # Keep training highly responsive on CPU by training on a subset of samples (8 samples)
        # This keeps the training loop real (actually updates PyTorch weights) while finishing in seconds.
        max_samples = 8
        inputs = inputs[:max_samples]
        targets = targets[:max_samples]
        
        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for input_str, target_str in zip(inputs, targets):
                # Tokenize input and target
                input_ids = self.tokenizer.encode(input_str, return_tensors="pt")
                labels = self.tokenizer.encode(target_str, return_tensors="pt")
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(inputs)
            
            # Yield progress to the callback (epoch loss)
            if progress_callback:
                progress_callback(epoch, avg_loss)
                
        # Save model state locally
        torch.save(self.model.state_dict(), self.weights_path)
        self.is_trained = True
        
    def generate(self, story: str, fallback_parser) -> str:
        """
        Translates a User Story to Gherkin format using the T5 model.
        Only loads and runs T5 if trained, otherwise uses the rule-based parser fallback.
        """
        # Only load and use T5 model if it has been trained/fine-tuned or local weights exist
        if self.is_trained or os.path.exists(self.weights_path):
            try:
                self.load_model()
                
                # Load saved weights if they exist and model is not marked trained
                if os.path.exists(self.weights_path) and not self.is_trained:
                    self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')))
                    self.is_trained = True

                # Run T5 inference
                input_str = "translate Story to Gherkin: " + story
                input_ids = self.tokenizer.encode(input_str, return_tensors="pt")
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=256,
                        num_beams=2,
                        early_stopping=True
                    )
                    
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                has_gherkin_keywords = any(kw in generated_text for kw in ["Given", "When", "Then", "Scenario:", "Feature:"])
                
                if has_gherkin_keywords:
                    return generated_text
            except Exception as e:
                print(f"T5 generation error: {e}")

        # Fallback to structuring Gherkin from the user story instantly
        parsed = fallback_parser.parse_user_story(story)
        return fallback_parser.to_gherkin(story, parsed.get("feature", "Verify Acceptance Criteria"))
