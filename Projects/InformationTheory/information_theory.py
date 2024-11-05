"""
Information Theory Lab

Matthew Mella
1
10/03/2023
"""


import numpy as np
import wordle

# Problem 1
def get_guess_result(guess, true_word):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the secret word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        guess (string) - the guess being made
        true_word (string) - the secret word
    Returns:
        result (list of integers) - the result of the guess, as described above
    """
    # Initialize the guess array with 0s and 2s
    guess_array = [2 * (guess[i] == true_word[i]) for i in range(len(guess))]

    # create a dictionary with the quantity of each letter in the true word
    true_word_dict = {a:true_word.count(a) for a in true_word}

    # loop through the guess word and decrease its count when the letter is in its correct location
    for i,a in enumerate(guess):
        if  guess_array[i] == 2:
            true_word_dict[a] -= 1
    
    # loop through the guess word and change the 0s to 1s when the letter is in the true word
    for i,a in enumerate(guess):
        if a in true_word and true_word_dict[a] > 0:
            guess_array[i] = 1
            true_word_dict[a] -= 1

    # return the guess array
    return guess_array


# Helper function
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
# Problem 2
def compute_highest_entropy(all_guess_results, allowed_guesses):
    """
    Compute the entropy of each allowed guess.
    
    Arguments:
        all_guess_results ((n,m) ndarray) - the array found in
            all_guess_results.npy, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_guesses (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
    """
    # Initialize the entropy array
    entropy_arr = []

    # Loop through the all_guess_results array and calculate the entropy for each guess
    for arr in all_guess_results:
        # Get the unique values and their counts
        unique_arr = np.unique(arr, return_counts=True)
        sum = 0
        total = np.sum(unique_arr[1])
        # Calculate the entropy
        for x in unique_arr[1]:
            p_x = x / total
            sum += p_x * np.log2(p_x)
        entropy_arr.append(-sum)

    # Return the guess with the highest entropy
    return allowed_guesses[np.argmax(entropy_arr)]
    
# Problem 3
def filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already have an array of the result of all guesses for all possible words, 
    we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (2-D ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        allowed_guesses (list of str)
            The list of words we are allowed to guess
        possible_secret_words (list of str)
            The list of possible secret words
        guess (str)
            The guess we made
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (2-D ndarray) The filtered array of guess results
    """
    # Helper function, converts a list of integers to base 3
    def list_to_base_3(int_list):
        sum = 0
        for i,num in enumerate(int_list):
            sum += num * 3**i
        return sum
    
    # Get the index of the guess
    guess_index = allowed_guesses.index(guess)

    # Get the result of the guess in base 3
    base_3_result = list_to_base_3(result)

    # Get the mask of the possible secret words with the same result as the guess
    mask = all_guess_results[guess_index] == base_3_result

    # Return the possible secret words and the filtered array of guess results
    secret_words_remaining = [word for i, word in enumerate(possible_secret_words) if mask[i]]
    return secret_words_remaining, all_guess_results[:,mask]

# Problem 4
def play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)

    # Get the secret word
    word = game.word
    
    # Loop through the game until it is finished
    while not game.is_finished():
        # If there is only one possible secret word, guess it
        if len(possible_secret_words) == 1:
            guess = possible_secret_words[0]
        else:
            # Get a random guess
            guess = np.random.choice(allowed_guesses)
        
        # Make the guess and filter the possible secret words
        result, guess_quantity = game.make_guess(guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)

    # Return the number of guesses
    return guess_quantity

# Problem 5
def play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
   # Initialize the game
    game.start_game(word=word, display=display)

    # Get the secret word
    word = game.word
    
    # Loop through the game until it is finished
    while not game.is_finished():
        # If there is only one possible secret word, guess it
        if len(possible_secret_words) == 1:
            guess = possible_secret_words[0]
        else:
            # Get the guess with the highest entropy
            guess = compute_highest_entropy(all_guess_results, allowed_guesses)
        
        # Make the guess and filter the possible secret words
        result, guess_quantity = game.make_guess(guess)
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)

    # Return the number of guesses
    return guess_quantity

# Problem 6
def compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    # add the guess quantity to the naive and entropy lists for each game
    naive_guesses = [play_game_naive(wordle.WordleGame(), all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False) for i in range(n)]
    entropy_guesses = [play_game_entropy(wordle.WordleGame(), all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False) for i in range(n)]

    # Return the average guess quantity for each algorithm
    return np.mean(naive_guesses), np.mean(entropy_guesses)
    
if __name__ == "__main__":
    print(compare_algorithms(np.load('all_guess_results.npy'), load_words('possible_secret_words.txt'), load_words('allowed_guesses.txt')))