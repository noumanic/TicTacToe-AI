import tkinter as tk
from tkinter import messagebox, font
import copy
import time
import random
from collections import defaultdict
import heapq

class UltimateTicTacToe:
    def __init__(self):
        # Initialize the game board
        # The board is represented as a 3x3 grid of 3x3 small boards
        # Each position can be 'X', 'O', or None (empty)
        self.board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.small_board_status = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.active_small_board = None  # (row, col) or None if any board is valid
        self.game_over = False
        self.winner = None
        
    def is_valid_move(self, big_row, big_col, small_row, small_col):
        """Check if a move is valid based on game rules."""
        # Check if the game is already over
        if self.game_over:
            return False
            
        # Check if the position is already occupied
        if self.board[big_row][big_col][small_row][small_col] is not None:
            return False
            
        # Check if the small board is already won
        if self.small_board_status[big_row][big_col] is not None:
            return False
            
        # Check if we must play in a specific small board
        if self.active_small_board is not None:
            req_row, req_col = self.active_small_board
            if big_row != req_row or big_col != req_col:
                return False
                
        return True
        
    def make_move(self, big_row, big_col, small_row, small_col):
        """Make a move on the board if it's valid."""
        if not self.is_valid_move(big_row, big_col, small_row, small_col):
            return False
            
        # Place the mark
        self.board[big_row][big_col][small_row][small_col] = self.current_player
        
        # Update small board status
        self.check_small_board_winner(big_row, big_col)
        
        # Set the next active small board
        if self.small_board_status[small_row][small_col] is not None or self.is_small_board_full(small_row, small_col):
            # If the target small board is already won or full, player can choose any board
            self.active_small_board = None
        else:
            self.active_small_board = (small_row, small_col)
            
        # Check for a winner on the big board
        self.check_big_board_winner()
        
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
        
    def check_small_board_winner(self, big_row, big_col):
        """Check if there's a winner in a specific small board."""
        small_board = self.board[big_row][big_col]
        
        # Check rows
        for row in range(3):
            if small_board[row][0] == small_board[row][1] == small_board[row][2] and small_board[row][0] is not None:
                self.small_board_status[big_row][big_col] = small_board[row][0]
                return
                
        # Check columns
        for col in range(3):
            if small_board[0][col] == small_board[1][col] == small_board[2][col] and small_board[0][col] is not None:
                self.small_board_status[big_row][big_col] = small_board[0][col]
                return
                
        # Check diagonals
        if small_board[0][0] == small_board[1][1] == small_board[2][2] and small_board[0][0] is not None:
            self.small_board_status[big_row][big_col] = small_board[0][0]
            return
            
        if small_board[0][2] == small_board[1][1] == small_board[2][0] and small_board[0][2] is not None:
            self.small_board_status[big_row][big_col] = small_board[0][2]
            return
            
        # Check if the small board is full (tie)
        if self.is_small_board_full(big_row, big_col):
            self.small_board_status[big_row][big_col] = 'Tie'
            
    def is_small_board_full(self, big_row, big_col):
        """Check if a small board is full."""
        for row in range(3):
            for col in range(3):
                if self.board[big_row][big_col][row][col] is None:
                    return False
        return True
        
    def check_big_board_winner(self):
        """Check if there's a winner on the big board."""
        # Check rows
        for row in range(3):
            if (self.small_board_status[row][0] == self.small_board_status[row][1] == self.small_board_status[row][2] and
                self.small_board_status[row][0] is not None and self.small_board_status[row][0] != 'Tie'):
                self.game_over = True
                self.winner = self.small_board_status[row][0]
                return
                
        # Check columns
        for col in range(3):
            if (self.small_board_status[0][col] == self.small_board_status[1][col] == self.small_board_status[2][col] and
                self.small_board_status[0][col] is not None and self.small_board_status[0][col] != 'Tie'):
                self.game_over = True
                self.winner = self.small_board_status[0][col]
                return
                
        # Check diagonals
        if (self.small_board_status[0][0] == self.small_board_status[1][1] == self.small_board_status[2][2] and
            self.small_board_status[0][0] is not None and self.small_board_status[0][0] != 'Tie'):
            self.game_over = True
            self.winner = self.small_board_status[0][0]
            return
            
        if (self.small_board_status[0][2] == self.small_board_status[1][1] == self.small_board_status[2][0] and
            self.small_board_status[0][2] is not None and self.small_board_status[0][2] != 'Tie'):
            self.game_over = True
            self.winner = self.small_board_status[0][2]
            return
            
        # Check if the big board is full (tie)
        big_board_full = True
        for row in range(3):
            for col in range(3):
                if self.small_board_status[row][col] is None:
                    big_board_full = False
                    break
            if not big_board_full:
                break
                
        if big_board_full:
            self.game_over = True
            self.winner = 'Tie'
            
    def get_valid_moves(self):
        """Get all valid moves."""
        valid_moves = []
        
        if self.active_small_board is not None:
            # Must play in the active small board
            big_row, big_col = self.active_small_board
            if self.small_board_status[big_row][big_col] is None:  # Ensure the board isn't already won
                for small_row in range(3):
                    for small_col in range(3):
                        if self.board[big_row][big_col][small_row][small_col] is None:
                            valid_moves.append((big_row, big_col, small_row, small_col))
        else:
            # Can play in any small board that isn't already won
            for big_row in range(3):
                for big_col in range(3):
                    if self.small_board_status[big_row][big_col] is None:
                        for small_row in range(3):
                            for small_col in range(3):
                                if self.board[big_row][big_col][small_row][small_col] is None:
                                    valid_moves.append((big_row, big_col, small_row, small_col))
                                    
        return valid_moves
        
    def is_game_over(self):
        """Check if the game is over."""
        return self.game_over
        
    def get_winner(self):
        """Get the winner of the game."""
        return self.winner


class CSPSolver:
    def __init__(self, game):
        self.game = game
        self.domains = {}  # Available values for each variable
        self.constraints = defaultdict(list)  # Constraints between variables
        self.assignment = {}  # Current assignment of values to variables
        self.next_player = 'X'  # Player who will make the next move
        
    def initialize_csp(self):
        """Initialize the CSP variables, domains, and constraints."""
        # Reset domains and constraints
        self.domains = {}
        self.constraints = defaultdict(list)
        self.next_player = self.game.current_player
        
        # Create variables for each position on the board
        for big_row in range(3):
            for big_col in range(3):
                # Skip if small board is already won
                if self.game.small_board_status[big_row][big_col] is not None:
                    continue
                    
                for small_row in range(3):
                    for small_col in range(3):
                        var = (big_row, big_col, small_row, small_col)
                        
                        # Check if position is already filled
                        if self.game.board[big_row][big_col][small_row][small_col] is not None:
                            continue
                            
                        # Check if this is a valid move based on active small board
                        if self.game.active_small_board is not None:
                            if (big_row, big_col) != self.game.active_small_board:
                                continue
                                
                        # This is a valid position for the next move
                        self.domains[var] = {self.next_player}
                            
        # Add constraints between variables
        for var1 in self.domains:
            for var2 in self.domains:
                if var1 != var2:
                    self.constraints[var1].append((var2, self.constraint_function))
                    
    def revise(self, var1, var2, constraint_function):
        """AC-3 revise function for arc consistency."""
        revised = False
        for value1 in list(self.domains[var1]):
            # If no value in var2's domain satisfies the constraint with value1
            if not any(constraint_function(var1, value1, var2, value2) 
                       for value2 in self.domains[var2]):
                self.domains[var1].remove(value1)
                revised = True
        return revised
        
    def ac3(self):
        """Apply the AC-3 algorithm for constraint propagation."""
        # Create a queue of arcs (variable pairs with constraints)
        queue = [(var1, var2, func) for var1 in self.constraints 
                 for (var2, func) in self.constraints[var1]]
                 
        while queue:
            var1, var2, constraint_function = queue.pop(0)
            
            if self.revise(var1, var2, constraint_function):
                if not self.domains[var1]:
                    return False  # No solution possible
                    
                # Add neighbors of var1 back to the queue
                for neighbor, func in self.constraints[var1]:
                    if neighbor != var2:
                        queue.append((neighbor, var1, func))
                        
        return True
        
    def constraint_function(self, var1, value1, var2, value2):
        """Check if assigning value1 to var1 and value2 to var2 satisfies constraints."""
        big_row1, big_col1, small_row1, small_col1 = var1
        big_row2, big_col2, small_row2, small_col2 = var2
        
        # In Ultimate Tic-Tac-Toe, moves are made sequentially
        # Each position can only be filled once
        if var1 == var2 and value1 == value2:
            return False
            
        # The target small board for the next move must follow from the previous move
        if (big_row1, big_col1) == (small_row2, small_col2) or (big_row2, big_col2) == (small_row1, small_col1):
            return True
            
        return True  # Default case - no constraint violation
        
    def get_neighbors(self, var):
        """Get all variables that share constraints with var."""
        return [v for v, _ in self.constraints[var]]
        
    def forward_checking(self, var, value):
        """Apply forward checking after assigning var=value."""
        # Create a copy of the domains to restore if needed
        saved_domains = {v: set(self.domains[v]) for v in self.domains}
        
        # Apply the assignment
        self.domains[var] = {value}
        
        # Update domains of connected variables
        for neighbor, constraint_function in self.constraints[var]:
            if neighbor in self.assignment:
                continue  # Skip if neighbor is already assigned
                
            # Remove values that are inconsistent with var=value
            for neighbor_value in list(self.domains[neighbor]):
                if not constraint_function(var, value, neighbor, neighbor_value):
                    self.domains[neighbor].remove(neighbor_value)
                    
            if not self.domains[neighbor]:
                # No valid value left for neighbor, restore domains and fail
                self.domains = saved_domains
                return False
                
        # Return an empty inference (no new assignments were made)
        return {}
        
    def select_unassigned_variable(self):
        """Minimum Remaining Values (MRV) heuristic with degree tie-breaker."""
        if not self.domains:
            return None
            
        # Use MRV heuristic: choose the variable with the fewest values in its domain
        # Break ties using degree heuristic: choose the variable involved in the most constraints
        best_var = None
        min_domain_size = float('inf')
        max_degree = -1
        
        for var in self.domains:
            if var in self.assignment:
                continue  # Skip assigned variables
                
            domain_size = len(self.domains[var])
            degree = len(self.constraints[var])
            
            if domain_size < min_domain_size or (domain_size == min_domain_size and degree > max_degree):
                min_domain_size = domain_size
                max_degree = degree
                best_var = var
                
        return best_var
        
    def order_domain_values(self, var):
        """Order domain values by least constraining value heuristic."""
        if len(self.domains[var]) <= 1:
            return list(self.domains[var])
            
        # For each value, count how many values would be eliminated from neighbors' domains
        def count_conflicts(value):
            count = 0
            for neighbor, constraint_function in self.constraints[var]:
                if neighbor in self.assignment:
                    continue  # Skip assigned variables
                    
                # Count values in neighbor's domain that conflict with var=value
                for neighbor_value in self.domains[neighbor]:
                    if not constraint_function(var, value, neighbor, neighbor_value):
                        count += 1
                        
            return count
            
        # Return values sorted by increasing conflict count (least constraining first)
        return sorted(self.domains[var], key=count_conflicts)
        
    def find_best_move(self):
        """Find the best move using CSP techniques."""
        # Initialize the CSP
        self.initialize_csp()
        
        # If no valid moves, return None
        if not self.domains:
            return None
            
        # Apply constraint propagation with AC-3
        if not self.ac3():
            return None  # No solution possible
            
        # Select the best variable (position) to assign next
        var = self.select_unassigned_variable()
        if var is None:
            return None
            
        # Return the chosen move
        return var
        
    def backtracking_search(self):
        """Backtracking search algorithm with forward checking."""
        # Initialize the CSP
        self.initialize_csp()
        self.assignment = {}
        
        # Apply constraint propagation
        if not self.ac3():
            return None  # No solution possible
            
        # Start the search
        result = self.backtrack()
        return result
        
    def backtrack(self):
        """Recursive backtracking algorithm with forward checking."""
        # Check if assignment is complete
        if len(self.assignment) == len(self.domains):
            return self.assignment
            
        # Select unassigned variable
        var = self.select_unassigned_variable()
        if var is None:
            return self.assignment  # No more variables to assign
            
        # Try each value in the domain
        for value in self.order_domain_values(var):
            # Check if consistent with current assignment
            consistent = True
            for assigned_var, assigned_value in self.assignment.items():
                for constraint_func in self.constraints[var]:
                    if not constraint_func[0](var, value, assigned_var, assigned_value):
                        consistent = False
                        break
                if not consistent:
                    break
                    
            if consistent:
                # Add to assignment
                self.assignment[var] = value
                
                # Apply forward checking
                fc_result = self.forward_checking(var, value)
                if fc_result is not False:
                    # Continue with recursive search
                    result = self.backtrack()
                    if result is not None:
                        return result
                        
                # Remove from assignment and restore domains
                del self.assignment[var]
                
        return None
        
    def is_consistent(self, var, value):
        """Check if assigning var=value is consistent with current assignment."""
        for assigned_var, assigned_value in self.assignment.items():
            for neighbor, constraint_func in self.constraints[var]:
                if neighbor == assigned_var and not constraint_func(var, value, assigned_var, assigned_value):
                    return False
        return True
        
    def evaluate_move(self, move):
        """Evaluate a move based on strategic heuristics."""
        big_row, big_col, small_row, small_col = move
        score = 0
        
        # Simulate making the move
        game_copy = copy.deepcopy(self.game)
        player = game_copy.current_player
        game_copy.make_move(big_row, big_col, small_row, small_col)
        
        # Check if the move wins a small board
        if game_copy.small_board_status[big_row][big_col] == player:
            score += 10
            
            # Check if winning this small board wins the game
            if game_copy.winner == player:
                score += 1000
                
        # Strategic positions
        # Center of a small board is valuable
        if small_row == 1 and small_col == 1:
            score += 3
            
        # Center of the big board is valuable
        if big_row == 1 and big_col == 1:
            score += 5
            
        # Corners of a small board are good
        if (small_row, small_col) in [(0,0), (0,2), (2,0), (2,2)]:
            score += 2
            
        # Corners of the big board are valuable
        if (big_row, big_col) in [(0,0), (0,2), (2,0), (2,2)]:
            score += 4
            
        # If the move sends opponent to a won board, that's good!
        if game_copy.active_small_board is None:
            score += 5  # Sending to any board is good
        elif game_copy.small_board_status[small_row][small_col] is not None:
            score += 10  # Sending to a won board is very good
            
        # If the move blocks a potential win in a small board, that's good
        opponent = 'O' if player == 'X' else 'X'
        if self._would_create_win_opportunity(game_copy, big_row, big_col, small_row, small_col, opponent):
            score += 15
            
        return score
        
    def _would_create_win_opportunity(self, game, big_row, big_col, small_row, small_col, player):
        """Check if a position would create a win opportunity for the given player."""
        # Check row
        row_count = 0
        for i in range(3):
            if game.board[big_row][big_col][small_row][i] == player:
                row_count += 1
        if row_count == 2:
            return True
            
        # Check column
        col_count = 0
        for i in range(3):
            if game.board[big_row][big_col][i][small_col] == player:
                col_count += 1
        if col_count == 2:
            return True
            
        # Check diagonals
        if small_row == small_col:  # Main diagonal
            diag_count = 0
            for i in range(3):
                if game.board[big_row][big_col][i][i] == player:
                    diag_count += 1
            if diag_count == 2:
                return True
                
        if small_row + small_col == 2:  # Other diagonal
            diag_count = 0
            for i in range(3):
                if game.board[big_row][big_col][i][2-i] == player:
                    diag_count += 1
            if diag_count == 2:
                return True
                
        return False
        

class MinimaxAI:
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth
        
    def get_best_move(self):
        """Get the best move using minimax with alpha-beta pruning."""
        best_val = float('-inf')
        best_move = None
        
        valid_moves = self.game.get_valid_moves()
        for move in valid_moves:
            # Make a copy of the game
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            
            # Evaluate the move
            val = self.minimax(game_copy, 0, False, float('-inf'), float('inf'))
            
            if val > best_val:
                best_val = val
                best_move = move
                
        return best_move
        
    def minimax(self, game, depth, is_maximizing, alpha, beta):
        """Minimax algorithm with alpha-beta pruning."""
        # Terminal conditions
        if game.is_game_over() or depth == self.max_depth:
            return self.evaluate(game)
            
        valid_moves = game.get_valid_moves()
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                
                eval = self.minimax(game_copy, depth + 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                    
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                
                eval = self.minimax(game_copy, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                    
            return min_eval
            
    def evaluate(self, game):
        """Evaluate the board state."""
        # Simple evaluation function
        # Actual implementation would be more sophisticated
        if game.winner == 'X':
            return 100
        elif game.winner == 'O':
            return -100
        else:
            # Count small boards won
            x_boards = sum(1 for row in range(3) for col in range(3) if game.small_board_status[row][col] == 'X')
            o_boards = sum(1 for row in range(3) for col in range(3) if game.small_board_status[row][col] == 'O')
            return x_boards - o_boards
            

class HybridCSPMinimaxAI:
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth
        self.csp_solver = CSPSolver(game)
        
    def get_best_move(self):
        """Get the best move using a hybrid of CSP and minimax with alpha-beta pruning."""
        # First, use CSP to narrow down the search space
        self.csp_solver.game = self.game  # Update the game state in CSP solver
        self.csp_solver.initialize_csp()
        
        # If no valid moves, return None
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None
            
        # Apply constraint propagation with AC-3
        if not self.csp_solver.ac3():
            # If no solution from CSP, fall back to standard minimax
            return self._minimax_fallback()
            
        # Get the reduced set of candidate moves from CSP
        candidate_moves = []
        for var in self.csp_solver.domains:
            if len(self.csp_solver.domains[var]) > 0:
                candidate_moves.append(var)
                
        # If no candidates, fall back
        if not candidate_moves:
            return self._minimax_fallback()
            
        # Evaluate the candidate moves with minimax and pick the best
        best_val = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort candidates by initial evaluation to improve alpha-beta efficiency
        candidate_moves.sort(key=lambda move: self.csp_solver.evaluate_move(move), reverse=True)
        
        for move in candidate_moves:
            # Make a copy of the game
            game_copy = copy.deepcopy(self.game)
            game_copy.make_move(*move)
            
            # Use faster pruning for deeper minimax search
            val = self.minimax(game_copy, 0, False, alpha, beta)
            
            if val > best_val:
                best_val = val
                best_move = move
                
            alpha = max(alpha, val)
                
        return best_move if best_move else valid_moves[0]  # Fallback to first valid move
    
    def _minimax_fallback(self):
        """Fallback to standard minimax if CSP fails."""
        return MinimaxAI(self.game, max_depth=2).get_best_move()
        
    def minimax(self, game, depth, is_maximizing, alpha, beta):
        """Minimax algorithm with alpha-beta pruning."""
        # Terminal conditions
        if game.is_game_over() or depth == self.max_depth:
            return self.evaluate(game)
            
        # Use MRV (Minimum Remaining Values) heuristic to order moves
        valid_moves = game.get_valid_moves()
        
        # For deeper levels, limit branching factor
        if depth > 0 and len(valid_moves) > 5:
            # Quick pre-evaluation to pick most promising moves
            moves_with_scores = [(move, self._quick_evaluate(game, move)) for move in valid_moves]
            moves_with_scores.sort(key=lambda x: x[1], reverse=is_maximizing)
            valid_moves = [move for move, _ in moves_with_scores[:5]]  # Keep top 5 moves
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                
                eval = self.minimax(game_copy, depth + 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
                    
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                
                eval = self.minimax(game_copy, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
                    
            return min_eval
    
    def _quick_evaluate(self, game, move):
        """Quick evaluation function for move ordering."""
        game_copy = copy.deepcopy(game)
        player = game_copy.current_player
        big_row, big_col, small_row, small_col = move
        
        # Check if move wins a small board
        game_copy.make_move(*move)
        if game_copy.small_board_status[big_row][big_col] == player:
            return 100
            
        # Check strategic positions
        score = 0
        # Center positions
        if small_row == 1 and small_col == 1:
            score += 3
        if big_row == 1 and big_col == 1:
            score += 5
            
        # Corners
        if (small_row, small_col) in [(0,0), (0,2), (2,0), (2,2)]:
            score += 2
            
        return score
            
    def evaluate(self, game):
        """Advanced evaluation function combining CSP insights and board evaluation."""
        if game.winner == 'X':
            return 1000
        elif game.winner == 'O':
            return -1000
        elif game.winner == 'Tie':
            return 0
            
        player = 'X' if game.current_player == 'O' else 'O'  # Opponent of current player
        
        # Score based on small boards won
        score = 0
        for r in range(3):
            for c in range(3):
                if game.small_board_status[r][c] == 'X':
                    score += 100
                elif game.small_board_status[r][c] == 'O':
                    score -= 100
                    
        # Score based on potential wins in small boards
        for big_r in range(3):
            for big_c in range(3):
                if game.small_board_status[big_r][big_c] is not None:
                    continue  # Skip won boards
                    
                # Count X and O in each row, column and diagonal in this small board
                for i in range(3):  # Rows
                    x_count = sum(1 for j in range(3) if game.board[big_r][big_c][i][j] == 'X')
                    o_count = sum(1 for j in range(3) if game.board[big_r][big_c][i][j] == 'O')
                    if x_count == 2 and o_count == 0:
                        score += 10
                    elif o_count == 2 and x_count == 0:
                        score -= 10
                        
                for j in range(3):  # Columns
                    x_count = sum(1 for i in range(3) if game.board[big_r][big_c][i][j] == 'X')
                    o_count = sum(1 for i in range(3) if game.board[big_r][big_c][i][j] == 'O')
                    if x_count == 2 and o_count == 0:
                        score += 10
                    elif o_count == 2 and x_count == 0:
                        score -= 10
                        
                # Diagonals
                x_diag1 = sum(1 for i in range(3) if game.board[big_r][big_c][i][i] == 'X')
                o_diag1 = sum(1 for i in range(3) if game.board[big_r][big_c][i][i] == 'O')
                if x_diag1 == 2 and o_diag1 == 0:
                    score += 10
                elif o_diag1 == 2 and x_diag1 == 0:
                    score -= 10
                    
                x_diag2 = sum(1 for i in range(3) if game.board[big_r][big_c][i][2-i] == 'X')
                o_diag2 = sum(1 for i in range(3) if game.board[big_r][big_c][i][2-i] == 'O')
                if x_diag2 == 2 and o_diag2 == 0:
                    score += 10
                elif o_diag2 == 2 and x_diag2 == 0:
                    score -= 10
        
        # Strategic board control evaluation
        center_board = game.small_board_status[1][1]
        if center_board == 'X':
            score += 50
        elif center_board == 'O':
            score -= 50
            
        # Control of corners
        corner_boards = [game.small_board_status[i][j] for i, j in [(0,0), (0,2), (2,0), (2,2)]]
        score += 20 * sum(1 for board in corner_boards if board == 'X')
        score -= 20 * sum(1 for board in corner_boards if board == 'O')
        
        # Adjust score based on current player
        return score if game.current_player == 'X' else -score


class UltimateTicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Tic-Tac-Toe")
        self.root.geometry("800x850")
        self.root.resizable(True, True)
        
        # Set up the game
        self.game = UltimateTicTacToe()
        
        # Set up AI (initially None)
        self.ai_type = None
        self.ai = None
        
        # Colors and styles
        self.bg_color = "#f0f0f0"
        self.grid_color = "#000000"
        self.active_color = "#a0e0a0"
        self.won_colors = {"X": "#ff9999", "O": "#9999ff", "Tie": "#dddddd"}
        
        # Fonts
        self.small_font = font.Font(family="Arial", size=14)
        self.medium_font = font.Font(family="Arial", size=18)
        self.large_font = font.Font(family="Arial", size=24, weight="bold")
        
        # Frame for the game board
        self.board_frame = tk.Frame(self.root, bg=self.bg_color)
        self.board_frame.pack(pady=20)
        
        # Canvas for drawing the game board
        self.canvas = tk.Canvas(self.board_frame, width=600, height=600, bg=self.bg_color)
        self.canvas.pack()
        
        # Status frame
        self.status_frame = tk.Frame(self.root, bg=self.bg_color)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.status_frame, 
            text="Player X's turn", 
            font=self.medium_font, 
            bg=self.bg_color
        )
        self.status_label.pack(pady=5)
        
        # Control frame
        self.control_frame = tk.Frame(self.root, bg=self.bg_color)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # AI selector
        self.ai_var = tk.StringVar(value="Human")
        self.ai_options = ["Human", "Minimax AI", "CSP-Minimax Hybrid AI"]
        
        self.ai_label = tk.Label(
            self.control_frame, 
            text="Player O:", 
            font=self.small_font, 
            bg=self.bg_color
        )
        self.ai_label.pack(side=tk.LEFT, padx=10)
        
        self.ai_menu = tk.OptionMenu(
            self.control_frame, 
            self.ai_var, 
            *self.ai_options, 
            command=self.set_ai
        )
        self.ai_menu.config(font=self.small_font)
        self.ai_menu.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_button = tk.Button(
            self.control_frame,
            text="New Game",
            font=self.small_font,
            command=self.reset_game
        )
        self.reset_button.pack(side=tk.RIGHT, padx=20)
        
        # Draw the initial board
        self.draw_board()
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.handle_click)
        
    def set_ai(self, selection):
        """Set the AI opponent."""
        self.ai_type = selection
        if selection == "Minimax AI":
            self.ai = MinimaxAI(self.game)
        elif selection == "CSP-Minimax Hybrid AI":
            self.ai = HybridCSPMinimaxAI(self.game)
        else:
            self.ai = None
        
    def reset_game(self):
        """Reset the game to the initial state."""
        self.game = UltimateTicTacToe()
        self.draw_board()
        self.status_label.config(text="Player X's turn")
        
    def draw_board(self):
        """Draw the game board."""
        self.canvas.delete("all")
        
        # Draw the big grid
        for i in range(1, 3):
            # Vertical lines
            self.canvas.create_line(
                i * 200, 0, i * 200, 600, 
                width=5, fill=self.grid_color
            )
            # Horizontal lines
            self.canvas.create_line(
                0, i * 200, 600, i * 200, 
                width=5, fill=self.grid_color
            )
            
        # Draw the small grids
        for big_row in range(3):
            for big_col in range(3):
                # Highlight active small board
                if (self.game.active_small_board is not None and 
                    self.game.active_small_board == (big_row, big_col) and
                    self.game.small_board_status[big_row][big_col] is None):
                    self.canvas.create_rectangle(
                        big_col * 200, big_row * 200,
                        (big_col + 1) * 200, (big_row + 1) * 200,
                        fill=self.active_color, outline=""
                    )
                
                # Draw small grid lines
                for i in range(1, 3):
                    # Vertical lines
                    self.canvas.create_line(
                        big_col * 200 + i * 66.67, big_row * 200,
                        big_col * 200 + i * 66.67, big_row * 200 + 200,
                        width=2, fill=self.grid_color
                    )
                    # Horizontal lines
                    self.canvas.create_line(
                        big_col * 200, big_row * 200 + i * 66.67,
                        big_col * 200 + 200, big_row * 200 + i * 66.67,
                        width=2, fill=self.grid_color
                    )
                    
                # Draw X's and O's in the small grid
                for small_row in range(3):
                    for small_col in range(3):
                        player = self.game.board[big_row][big_col][small_row][small_col]
                        if player is not None:
                            x = big_col * 200 + small_col * 66.67 + 33.33
                            y = big_row * 200 + small_row * 66.67 + 33.33
                            
                            if player == 'X':
                                self.canvas.create_line(
                                    x - 20, y - 20, x + 20, y + 20,
                                    width=3, fill="#ff0000"
                                )
                                self.canvas.create_line(
                                    x + 20, y - 20, x - 20, y + 20,
                                    width=3, fill="#ff0000"
                                )
                            elif player == 'O':
                                self.canvas.create_oval(
                                    x - 20, y - 20, x + 20, y + 20,
                                    width=3, outline="#0000ff"
                                )
                                
                # Draw small board status if it's won
                status = self.game.small_board_status[big_row][big_col]
                if status is not None:
                    color = self.won_colors.get(status, "#dddddd")
                    
                    # Draw semi-transparent overlay
                    self.canvas.create_rectangle(
                        big_col * 200, big_row * 200,
                        (big_col + 1) * 200, (big_row + 1) * 200,
                        fill=color, stipple="gray50"
                    )
                    
                    # Draw winner symbol
                    if status == 'X':
                        # Draw big X
                        self.canvas.create_line(
                            big_col * 200 + 50, big_row * 200 + 50,
                            big_col * 200 + 150, big_row * 200 + 150,
                            width=8, fill="#ff0000"
                        )
                        self.canvas.create_line(
                            big_col * 200 + 150, big_row * 200 + 50,
                            big_col * 200 + 50, big_row * 200 + 150,
                            width=8, fill="#ff0000"
                        )
                    elif status == 'O':
                        # Draw big O
                        self.canvas.create_oval(
                            big_col * 200 + 50, big_row * 200 + 50,
                            big_col * 200 + 150, big_row * 200 + 150,
                            width=8, outline="#0000ff"
                        )
                    elif status == 'Tie':
                        # Draw tie symbol
                        self.canvas.create_text(
                            big_col * 200 + 100, big_row * 200 + 100,
                            text="TIE", font=self.large_font,
                            fill="#555555"
                        )
        
        # Update game status message
        if self.game.game_over:
            if self.game.winner == 'Tie':
                self.status_label.config(text="Game Over: It's a Tie!")
            else:
                self.status_label.config(text=f"Game Over: Player {self.game.winner} Wins!")
        else:
            active_text = ""
            if self.game.active_small_board is not None:
                row, col = self.game.active_small_board
                active_text = f" (playing in board [{row},{col}])"
            self.status_label.config(text=f"Player {self.game.current_player}'s turn{active_text}")
                
    def handle_click(self, event):
        """Handle mouse click event."""
        if self.game.game_over:
            return
            
        # Ignore click if it's AI's turn
        if self.game.current_player == 'O' and self.ai is not None:
            return
            
        # Convert click coordinates to board position
        big_col = event.x // 200
        big_row = event.y // 200
        
        small_col = (event.x % 200) // 67
        small_row = (event.y % 200) // 67
        
        # Make the move
        if self.game.make_move(big_row, big_col, small_row, small_col):
            self.draw_board()
            
            # If the game isn't over and it's AI's turn, make an AI move
            if not self.game.game_over and self.game.current_player == 'O' and self.ai is not None:
                self.root.after(500, self.make_ai_move)
                
    def make_ai_move(self):
        """Make a move using the selected AI."""
        if self.game.game_over:
            return
            
        # Get the best move from the AI
        move = self.ai.get_best_move()
        
        if move:
            # Make the move
            self.game.make_move(*move)
            self.draw_board()


def main():
    """Run the game."""
    root = tk.Tk()
    app = UltimateTicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()