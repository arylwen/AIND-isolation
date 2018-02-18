"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

#penalize the moves farther from the center
def centrality_measure(game, move):
    x, y = move
    center_x, center_y = (math.ceil(game.width/2.), math.ceil(game.height/2.))
    distance_from_center = (game.width - center_x)**2 + (game.height - center_y)**2 - ((x-center_x)**2 + (y-center_y)**2)
    return distance_from_center      

def find_common_moves(game, player):
    player_moves = game.get_legal_moves()
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    return player_moves and opponent_moves

#penalize common moves
def common_moves_measure(game, player):
    common_moves = find_common_moves(game, player)
    return 8 - len(common_moves)

#common move closest the center
def interfering_moves_measure(game, player):
    common_moves = find_common_moves(game, player)
    if not common_moves:
        return 0
    return max(centrality_measure(game, m) for m in common_moves)  

"""
Evaluation function yelding a score equal with the difference in moves
available to the two players to which we add a measure on how close the player
is to the center
"""
def added_centrality_heuristic(game, player):
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)
    player_moves = len(game.get_legal_moves())
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves - opponent_moves + 
        centrality_measure(game, game.get_player_location(player)) )  

"""
encourage moves with available moves closer to center, less common moves and common moves closer to the cente 
"""
def added_centrality_sum_heuristic(game, player):
    opponent = game.get_opponent(player)
    opponent_moves = game.get_legal_moves(opponent)
    player_moves = game.get_legal_moves()
    if not opponent_moves:
        return float("inf")
    if not player_moves:
        return float("-inf")
    return float(len(player_moves) - len(opponent_moves) + 
        sum(centrality_measure(game, m) for m in player_moves) + 
        common_moves_measure(game, player) + 
        interfering_moves_measure(game, player))

"""
 penalize common moves, encourage available moves
"""
def weighted_common_moves_heuristic(game, player):

    opponent = game.get_opponent(player)
    opponent_moves = game.get_legal_moves(opponent)
    player_moves = game.get_legal_moves()
    common_moves = opponent_moves and player_moves
    if not opponent_moves:
        return float("inf")
    if not player_moves:
        return float("-inf")
    #decay factor as the game goes on
    weight_factor = 1/(game.move_count +1 )
    inverse_weight_factor = 1/weight_factor
    #penalize common moves, encourage available moves
    return float(len(common_moves)*weight_factor 
        + inverse_weight_factor*len(game.get_legal_moves()))

def weighted_centrality_heuristic(game, player):
    '''
    opponent_weight - avoid choices where the opponent has more moves
    center_weight - favorize the center more
    '''   
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    central_column = math.ceil(game.width/2.)
    central_row = math.ceil(game.height/2.)
    num_all_available_moves = float(game.width * game.height)
    num_blank_spaces = len(game.get_blank_spaces())
    decay = num_blank_spaces/num_all_available_moves

    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    num_player_moves = len(player_moves)
    num_opponent_moves = len( opponent_moves )

    player_weight = 1
    opponent_weight = 2
    center_weight = 2

    for move in player_moves:
        if move[0] == central_row or move[1] == central_column:
            player_weight *= (center_weight * decay)

    for move in opponent_moves:
        if move[0] == central_row or move[1] == central_column:  
            opponent_weight  *= (center_weight * decay)

    return float((num_player_moves * player_weight) - 
        (num_opponent_moves * opponent_weight))            

def sun_tzu_heuristic(game, player):

    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    dispersive_ground_weight = 2.0
    facile_ground_weight = 4.0
    contentious_ground_weight = 2.0
    open_ground_weight = 2.0
    serious_ground_weight = 8.0
    difficult_ground_weight = 4.0
    hemmed_in_ground_weight = 8.0

    central_column = math.ceil(game.width/2.)
    central_row = math.ceil(game.height/2.)
    num_all_available_moves = float(game.width * game.height)
    num_blank_spaces = len(game.get_blank_spaces())
    decay = num_blank_spaces/num_all_available_moves

    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    num_player_moves = len(player_moves)
    num_opponent_moves = len( opponent_moves )
    common_moves = len(find_common_moves(game, player))

    dispersive_ground = num_opponent_moves/num_player_moves if num_player_moves !=0 else num_opponent_moves
    facile_ground = num_player_moves/num_opponent_moves if num_opponent_moves !=0 else num_player_moves
    contentious_ground = common_moves/num_blank_spaces if common_moves !=0 else 1.
    serious_ground = num_player_moves/num_opponent_moves if num_opponent_moves !=0 else num_player_moves
    difficult_ground = num_opponent_moves/num_player_moves if num_player_moves !=0 else num_opponent_moves
    hemmed_in_ground = num_opponent_moves/num_player_moves if num_player_moves !=0 else num_opponent_moves

    dispersive_ground = dispersive_ground if dispersive_ground !=0 else 1
    facile_ground = facile_ground if facile_ground !=0 else 1
    serious_ground = serious_ground if serious_ground != 0 else 1
    difficult_ground = difficult_ground if difficult_ground !=0 else 1
    hemmed_in_ground = hemmed_in_ground if hemmed_in_ground !=0 else 1

    player_weight=1.0
    opponent_weight = 2.0

    #handle open_ground/center moves
    for move in player_moves:
        if move[0] == central_row or move[1] == central_column:
            player_weight *= (open_ground_weight * decay)

    for move in opponent_moves:
        if move[0] == central_row or move[1] == central_column:  
            opponent_weight  *= (open_ground_weight * decay)

    #factor ground weights only once
    player_weight *= facile_ground_weight*serious_ground_weight
    opponent_weight *= dispersive_ground_weight*contentious_ground_weight*difficult_ground_weight*hemmed_in_ground_weight

    player_weight *= (decay * facile_ground) * (decay * serious_ground)
    opponent_weight *= (decay * dispersive_ground) * (decay * contentious_ground) * (decay * difficult_ground) * (decay * hemmed_in_ground)

    return float((num_player_moves * player_weight) - 
        (num_opponent_moves * opponent_weight))            



def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return added_centrality_heuristic(game, player)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return weighted_centrality_heuristic(game, player)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return sun_tzu_heuristic(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def is_active_player(self, game):
        """
        """
        return game.active_player == self

    def make_a_move(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return (game.get_player_location(self), self.score(game, self)) 

        value, func, best_move = None, None, (-1, -1)

        if self.is_active_player(game):
            func, value = max, float("-inf")
        else:
            func, value = min, float("inf")       

        for move in game.get_legal_moves():
            next_move = game.forecast_move(move)
            score = self.make_a_move(next_move, depth-1)[1]
            if func(value, score) == score:
                best_move = move
                value = score

        return (best_move, value)        

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        return self.make_a_move(game, depth)[0] 


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        #self.time_left = time_left

        # TODO: finish this function!
        self.time_left = time_left
        best_move = (-1, -1)
        for i in range(1, game.width * game.height):
            #score, best_move = self.alphabeta(game, i)
            try:
                #print("before make_a_move")
                #best_move, score = self.make_a_move(game, i)
                best_move = self.alphabeta(game, i)
                #if score == float('inf'):
                #    break
            except SearchTimeout:
                #print("SearchTimeout")
                return best_move    
        
        return best_move        

    def is_active_player (self, game):
        return game.active_player == self
                         
    def make_a_move(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # at depth 0 we did not move yet
        if depth== 0: 
            return self.score(game, self), (-1, -1)

        # legal moves for active player
        legal_moves= game.get_legal_moves(game.active_player)

        # handle terminal game state
        if not legal_moves: 
            return self.score(game, self), (-1, -1)

        # no best_move yet
        best_action= None
        # recursive call depending on player
        if self.is_active_player(game): #"us"
            best_score= float('-inf') # classic max alg - start the lowest
            for move in legal_moves:
                next_play= game.forecast_move(move)
                # opponent will try to lower (min) our scores
                score = self.make_a_move(next_play, depth - 1, alpha, beta)[0]
                # make the best of it - pick the highest - classic max
                if score > best_score:
                    best_score= score
                    best_action= move
                # new alpha is best of past and present
                alpha= max(alpha, score)
                if score >= beta: 
                    break #high enough to maybe win


        else: # opponent 
            best_score= float('inf') # classic max alg - start the highest
            for move in legal_moves:
                next_play= game.forecast_move(move)
                # opponent trying to guess our best (max) move
                score = self.make_a_move(next_play, depth - 1,alpha, beta)[0]
                # they're trying to beat us and lower our score - pick the lowest
                if score < best_score:
                    best_score= score
                    best_action= move
                # assume the worst
                beta= min(beta, score)
                if score <= alpha: 
                    break #low enough to create problems

        return best_score, best_action

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #print("alphabeta")
        # TODO: finish this function!
        return self.make_a_move(game, depth)[1]

