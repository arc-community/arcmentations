import numpy as np
from arc.interface import BoardPair, Board

def cropInputAndOutput(inputPair:BoardPair, cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row)->BoardPair:
    input_np_board = inputPair.input.np
    output_np_board = inputPair.output.np
    input_np_board = cropBoard(input_np_board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row).np
    output_np_board = cropBoard(output_np_board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row).np

    return BoardPair(input=input_np_board.tolist(),output=output_np_board.tolist())

def cropBoard(boardIn:Board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row)->Board:
    if cols_to_rem_choice > 0 : 
        if dir_choice_col == -1:
            boardIn = boardIn[cols_to_rem_choice:,:]
        else:
            boardIn = boardIn[:-cols_to_rem_choice,:]
    if rows_to_rem_choice > 0:
        if dir_choice_row == -1:
            boardIn = boardIn[:,rows_to_rem_choice:]
        else:
            boardIn = boardIn[:,:-rows_to_rem_choice]

    return Board(__root__ = boardIn.tolist())

def doubleBoard(boardIn:Board,separation:int,is_horizontal_cat:bool,z_index_of_original:bool=0)->Board:
    """
    This function takes a Board and returns a new Board with the input board doubled.
    If is_horizontal_cat is True, the board is horizontally concatenated otherwise concatenation is vertical.
    Separation is the number of pixels between the two boards.
    This function supports negative separation values. In this case, one board is cropped to accomodate the other.
    if separation is negative, the z_index_of_original is used to determine which board lays ontop of the other.
    """

    np_board = boardIn.np
    ret_board_size = [len(np_board),len(np_board[0])]
    if is_horizontal_cat:
        ret_board_size[1] = ret_board_size[1]*2 + separation
    else:
        ret_board_size[0] = ret_board_size[0]*2 + separation
    ret_board = np.zeros(ret_board_size)

    if is_horizontal_cat:
        if z_index_of_original == 0:
            ret_board[:,:len(np_board[0])] = np_board
            ret_board[:,-len(np_board[0]):] = np_board
        else:
            ret_board[:,-len(np_board[0]):] = np_board
            ret_board[:,:len(np_board[0])] = np_board

    else:
        if z_index_of_original == 0:
            ret_board[:len(np_board),:] = np_board
            ret_board[-len(np_board):,:] = np_board
        else:
            ret_board[-len(np_board):,:] = np_board
            ret_board[:len(np_board),:] = np_board

    return Board(__root__ = ret_board.tolist())

def doubleInputBoard(board_pair:BoardPair,separation:int,is_horizontal_cat:bool,z_index_of_original:bool=0)->BoardPair:
    board_pair.input = doubleBoard(board_pair.input,separation,is_horizontal_cat,z_index_of_original)
    return board_pair
