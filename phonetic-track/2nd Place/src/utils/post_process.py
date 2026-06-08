


class PostProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def post_process_predictions(self, predictions):
        # Example post-processing: remove duplicates and apply heuristic rules
        processed = []
        for pred in predictions:
            # Remove duplicates
            unique_pred = self.remove_duplicates(pred)
            
            processed.append(unique_pred)
        return processed


    def remove_duplicates(self, s):
        if len(s) < 3:
            return s
        result = []
        prev_char = None
        for char in s:
            if char != prev_char:
                result.append(char)
            prev_char = char
        return ''.join(result)