import ctypes
import tempfile
import os
import io
import sys
import subprocess
import random

import tkinter as tk
from tkinter import messagebox




def print_board(board, n):
    print([f" {i}" for i in range(n)])
    for i in range(n):
        print(" " * i, end="")
        print(i,[ "🟥"if board[i][j] == 2 else "🟦" if board[i][j] == 1 else "⬜️" for j in range(n)])
        print("")
    print()


class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self) -> 'HexBoard':
        """Devuelve una copia del tablero actual"""
        pass

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        pass

    def get_possible_moves(self) -> list:
        """Devuelve todas las casillas vacías como tuplas (fila, columna)."""
        pass
    
    def check_connection(self, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados"""
        pass

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("Implement this method!")

class AstroBot(Player):
    def __init__(self, player_id: int):

        with open("c_code.c", "r") as f:
            c_code = f.read()

        self.player_id = player_id
        self.num_play = 0

        with open("c_code.c", "r") as f:
            self.c_code = f.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the C code to a temporary file
            c_file = os.path.join(tmpdir, "temp.c")
            so_file = os.path.join(tmpdir, "temp.so")
            
            with open(c_file, "w") as file:
                file.write(c_code)
            
            # Compile as shared library
            os.system(f"gcc -shared -fPIC {c_file} -o {so_file}")
            
            # Load the shared library
            self.lib = ctypes.CDLL(so_file)

        self.lib.process_hex_board.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # Pointer to a 2D array
            ctypes.c_int,  # Board size (n)
            ctypes.c_int,  # Depth
            ctypes.c_int,  # Player
            ctypes.c_longlong,  # Wait time
            ctypes.POINTER(ctypes.c_int),  # Best move row
            ctypes.POINTER(ctypes.c_int)   # Best move column
        ]
        
        self.lib.process_hex_board.restype = ctypes.c_float  # Define return type

    def play(self, time: int, board: 'HexBoard') -> tuple:
        """Choose a move to play using the C library."""
        if self.num_play == 0 and self.player_id == 1:
            print("First move for player 1")
            # First move for player 1
            row = board.size // 2
            col = board.size // 2
            self.num_play += 1
            return row, col
        if self.num_play == 0 and self.player_id == 2:
            print("First move for player 1")
            # First move for player 2
            
            row = board.size // 2
            col = board.size // 2
            self.num_play += 1
            if(board.board[row][col]==0):
                return row, col
            else:
                return row, col+1
            
        
        self.num_play += 1

        depth = 4

        n = board.size
        c_board = (ctypes.POINTER(ctypes.c_int) * n)()
        for i in range(n):
            row = (ctypes.c_int * n)()
            for j in range(n):
                row[j] = board.board[i][j]
            c_board[i] = row

        best_i = ctypes.c_int(-1)
        best_j = ctypes.c_int(-1)

        player = self.player_id
        wait_time = time

        result = self.lib.process_hex_board(c_board, n, depth, player, wait_time, ctypes.byref(best_i), ctypes.byref(best_j))

        for i in range(n):
            ctypes.cast(c_board[i], ctypes.POINTER(ctypes.c_int))

        if best_i.value != -1 and best_j.value != -1:
            return best_i.value, best_j.value
        return None



class Hex(HexBoard):
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self) -> 'HexBoard':
        """Return a copy of the current board."""
        new_board = HexBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Place a piece if the cell is empty."""
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True
        return False

    def get_possible_moves(self) -> list:
        """Return all empty cells as tuples (row, column)."""
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]

    def check_connection(self, player_id: int) -> bool:
        """Check if the player has connected their two sides."""
        visited = set()

        def dfs(row, col):
            if (row, col) in visited:
                return False
            visited.add((row, col))
            if player_id == 1 and col == self.size - 1: 
                return True
            if player_id == 2 and row == self.size - 1:
                return True
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player_id:
                    if dfs(nr, nc):
                        return True
            return False

        if player_id == 1: 
            for row in range(self.size):
                if self.board[row][0] == player_id and dfs(row, 0):
                    return True
        elif player_id == 2:
            for col in range(self.size):
                if self.board[0][col] == player_id and dfs(0, col):
                    return True
        return False

