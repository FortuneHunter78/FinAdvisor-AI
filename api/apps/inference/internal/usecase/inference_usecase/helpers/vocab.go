package helpers

import (
	"bufio"
	"errors"
	"os"
	"strings"
)

// Vocabulary map to store the word-to-ID mapping
var vocab map[string]int

type TokenizedOutput struct {
	InputWordIDs [][]int32 `json:"input_word_ids"`
	InputMask    [][]int32 `json:"input_mask"`
	InputTypeIDs [][]int32 `json:"input_type_ids"`
}

// LoadVocabulary loads the vocabulary from the file
func LoadVocabulary(vocabFile string) error {
	vocab = make(map[string]int)
	file, err := os.Open(vocabFile)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	idx := 0
	for scanner.Scan() {
		token := scanner.Text()
		vocab[token] = idx
		idx++
	}

	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

// Basic word tokenizer (you can improve this)
func Tokenize(text string) []string {
	// Convert text to lowercase
	text = strings.ToLower(text)

	// Split text by spaces
	tokens := strings.Fields(text)

	return tokens
}

// WordPiece tokenization
func WordPieceTokenize(word string) []string {
	// Start with the full word
	if _, exists := vocab[word]; exists {
		// If the word exists in the vocabulary, return it as is
		return []string{word}
	}

	// Try splitting the word into smaller subwords
	subwords := []string{}
	for i := 1; i <= len(word); i++ {
		subword := word[:i]
		if _, exists := vocab[subword]; exists {
			subwords = append(subwords, subword)
			// Try to tokenize the remaining part
			remaining := word[i:]
			subwords = append(subwords, WordPieceTokenize(remaining)...)
			break
		}
	}
	return subwords
}

// Tokenize the input text with WordPiece subword tokenization
func TokenizeWithWordPiece(text string) []string {
	tokens := Tokenize(text)
	var wordPieceTokens []string

	for _, token := range tokens {
		if _, exists := vocab[token]; exists {
			wordPieceTokens = append(wordPieceTokens, token)
		} else {
			subwords := WordPieceTokenize(token)
			wordPieceTokens = append(wordPieceTokens, subwords...)
		}
	}

	return wordPieceTokens
}

// Convert tokens to token IDs
func TokensToIds(tokens []string) []int {
	var ids []int
	for _, token := range tokens {
		if id, exists := vocab[token]; exists {
			ids = append(ids, id)
		} else {
			// Handle unknown tokens (e.g., "[UNK]")
			unkID, exists := vocab["[UNK]"]
			if exists {
				ids = append(ids, unkID)
			} else {
				ids = append(ids, 0) // Default to 0 if [UNK] is missing
			}
		}
	}
	return ids
}

// Tokenize the input text using WordPiece
func TokenizeText(text string) (TokenizedOutput, error) {
	if vocab == nil {
		return TokenizedOutput{}, errors.New("vocabulary is not loaded")
	}

	// Tokenize input
	tokens := TokenizeWithWordPiece(text)

	// Convert tokens to IDs
	tokenIDs := TokensToIds(tokens)

	// Add [CLS] and [SEP] tokens (assuming they exist in vocab)
	clsID, clsExists := vocab["[CLS]"]
	sepID, sepExists := vocab["[SEP]"]

	if !clsExists || !sepExists {
		return TokenizedOutput{}, errors.New("missing special tokens ([CLS] or [SEP]) in vocab")
	}

	tokenIDs = append([]int{clsID}, tokenIDs...) // Prepend [CLS]
	tokenIDs = append(tokenIDs, sepID)           // Append [SEP]

	// Convert to int32 slices
	inputWordIDs := [][]int32{convertToInt32(tokenIDs)}
	inputMask := make([][]int32, 1)
	inputTypeIDs := make([][]int32, 1)

	// Generate attention mask (1 for real tokens, 0 for padding)
	for range tokenIDs {
		inputMask[0] = append(inputMask[0], 1)
		inputTypeIDs[0] = append(inputTypeIDs[0], 0) // Assuming single-sequence input
	}

	return TokenizedOutput{
		InputWordIDs: inputWordIDs,
		InputMask:    inputMask,
		InputTypeIDs: inputTypeIDs,
	}, nil
}

// Convert int slice to int32 slice
func convertToInt32(arr []int) []int32 {
	var int32Arr []int32
	for _, val := range arr {
		int32Arr = append(int32Arr, int32(val))
	}
	return int32Arr
}
