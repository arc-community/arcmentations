from enum import Enum
import numpy as np
from arc.interface import BoardPair, Board, Riddle
import imutils
import scipy
import skimage


class Direction(Enum):
    horizontal = 1
    vertical = 2
    both = 3


def noop(o):
    return o


def cropInputAndOutput(
    inputPair: BoardPair,
    cols_to_rem_choice,
    rows_to_rem_choice,
    dir_choice_col,
    dir_choice_row,
) -> BoardPair:
    input_np_board = inputPair.input.np
    output_np_board = inputPair.output.np
    input_np_board = cropBoard(
        input_np_board,
        cols_to_rem_choice,
        rows_to_rem_choice,
        dir_choice_col,
        dir_choice_row,
    ).np
    output_np_board = cropBoard(
        output_np_board,
        cols_to_rem_choice,
        rows_to_rem_choice,
        dir_choice_col,
        dir_choice_row,
    ).np

    return BoardPair(input=input_np_board.tolist(), output=output_np_board.tolist())


def cropInputOnly(
    inputPair: BoardPair,
    cols_to_rem_choice,
    rows_to_rem_choice,
    dir_choice_col,
    dir_choice_row,
) -> BoardPair:
    input_np_board = inputPair.input.np
    input_np_board = cropBoard(
        input_np_board,
        cols_to_rem_choice,
        rows_to_rem_choice,
        dir_choice_col,
        dir_choice_row,
    ).np

    return BoardPair(input=input_np_board.tolist(), output=inputPair.output)


def cropBoard(
    boardIn: np.ndarray,
    cols_to_rem_choice,
    rows_to_rem_choice,
    dir_choice_col,
    dir_choice_row,
) -> Board:
    if cols_to_rem_choice > 0:
        if dir_choice_col == -1:
            boardIn = boardIn[cols_to_rem_choice:, :]
        else:
            boardIn = boardIn[:-cols_to_rem_choice, :]
    if rows_to_rem_choice > 0:
        if dir_choice_row == -1:
            boardIn = boardIn[:, rows_to_rem_choice:]
        else:
            boardIn = boardIn[:, :-rows_to_rem_choice]

    return Board(__root__=boardIn.tolist())


def floatRotateAll(board_pair: BoardPair, angle: int, superres_scale_fac) -> BoardPair:
    board_pair.input = floatRotate(board_pair.input, angle, superres_scale_fac)
    board_pair.output = floatRotate(board_pair.output, angle, superres_scale_fac)
    return board_pair


def floatRotateAll2(r: Riddle, angle: int) -> Riddle:
    # case 1 all inputs and outputs are the same size
    # case 2 all outputs are the same size, all inputs are the same size
    # case 3 inputs have the same size as outputs
    # case 4 no known size relationship
    # if the sizes are meant to be equal in all including test then make them equal.
    for board_pair in r.train:
        board_pair.input = customFloatRotate(board_pair.input, angle)
        board_pair.output = customFloatRotate(board_pair.output, angle)
    for board_pair in r.test:
        board_pair.input = customFloatRotate(board_pair.input, angle)
        board_pair.output = customFloatRotate(board_pair.output, angle)
    return r


def floatRotate(board_in: Board, angle: int, superres_scale_fac) -> Board:
    # board_in = Board(__root__= np.flip(board_in.np, 1).tolist())
    board_in_np_border = np.pad(board_in.np, ((1, 1), (1, 1)), constant_values=0)
    board_in_np_border = Board(__root__=board_in_np_border.tolist())
    superres_scale = 10
    upscaled_factor_2_board = superResolutionBoard(
        board_in_np_border, superres_scale, Direction.both
    )
    board_in_scaled_np = upscaled_factor_2_board.np
    board_in_separated = np.eye(10)[board_in_scaled_np]
    board_in_rotated = scipy.ndimage.rotate(
        board_in_separated, angle, order=0, prefilter=False
    )

    # convolve the board with 10x10 kernel of ones, stride of 10, so that we downscale the board by 10
    def func(block, axis):
        return np.mean(block, axis=axis)
        return np.mean(block[:, :, :, 3:7, 3:7, :], axis=axis)

    upscale_fac = superres_scale_fac
    downscaled_board = skimage.measure.block_reduce(
        board_in_rotated,
        (superres_scale // upscale_fac, superres_scale // upscale_fac, 1),
        func=func,
    )
    downscaled_board[:, :, 0] = downscaled_board[:, :, 0] / 2.1
    ret = np.argmax(downscaled_board, -1)
    # ret = scipy.ndimage.rotate(board_in.np, angle, order=0, prefilter=False)
    return Board(__root__=ret.tolist())


def customFloatRotate(board_in: Board, angle: int) -> Board:
    # assert board_in.np.shape[0] % 2 == 1 and board_in.np.shape[1] % 2 == 1
    center_y, center_x = np.array(board_in.np.shape) / 2

    # Create a rotation matrix using the given angle
    angle_rad = np.deg2rad(angle)
    cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    rotated_points = []

    min_x, min_y = 100000, 100000
    max_x, max_y = -100000, -100000
    # find all 4 corners points
    corners = [
        (0, 0),
        (0, len(board_in.np[0]) - 1),
        (len(board_in.np) - 1, 0),
        (len(board_in.np) - 1, len(board_in.np[0]) - 1),
    ]
    for y, x in corners:
        trans_point = np.array([y - center_y, x - center_x])
        rotated_point = rotation_matrix @ trans_point
        rotated_point[0] += center_y
        rotated_point[1] += center_x
        min_x, max_x = min(min_x, rotated_point[1]), max(max_x, rotated_point[1])
        min_y, max_y = min(min_y, rotated_point[0]), max(max_y, rotated_point[0])

    for y in range(len(board_in.np)):
        for x in range(len(board_in.np[0])):
            if board_in.np[y][x] != 0:
                # Translate points to origin for rotation
                trans_point = np.array([y - center_y, x - center_x])
                # Apply rotation
                rotated_point = rotation_matrix @ trans_point
                # Translate back from origin
                rotated_point[0] += center_y
                rotated_point[1] += center_x

                rotated_points.append((rotated_point, board_in.np[y][x]))

                # min_x, max_x = min(min_x, rotated_point[1]), max(max_x, rotated_point[1])
                # min_y, max_y = min(min_y, rotated_point[0]), max(max_y, rotated_point[0])

    # Determine new board size based on the bounding box
    # new_height = int(np.ceil(max_y ) - np.sign(min_y)*np.ceil(abs(min_y)))
    # new_width = int(np.ceil(max_x  ) - np.sign(min_x)*np.ceil(abs(min_x)))
    new_height = int(np.ceil(max_y) - np.floor(min_y))
    new_width = int(np.ceil(max_x) - np.floor(min_x))
    # Create a new board with the determined size
    new_board_array = np.zeros((new_height, new_width), dtype=board_in.np.dtype)

    # Offset to translate rotated points to the new board's coordinate space
    offset_y, offset_x = -min_y, -min_x

    deduped_points = []

    def key(x):
        last_decimal_x = abs(x[0][0] - int(x[0][0]))
        last_decimal_y = abs(x[0][1] - int(x[0][1]))
        if last_decimal_x > 0.5:
            last_decimal_x = 1 - last_decimal_x
        if last_decimal_y > 0.5:
            last_decimal_y = 1 - last_decimal_y
        return last_decimal_x + last_decimal_y

    rotated_points = sorted(rotated_points, key=key)
    dedup_set = set()
    for point, color in rotated_points:
        if (int(point[0] + offset_y), int(point[1] + offset_x)) not in dedup_set:
            dedup_set.add((int(point[0] + offset_y), int(point[1] + offset_x)))
            deduped_points.append(
                (int(point[0] + offset_y), int(point[1] + offset_x), color)
            )
        else:
            if (
                int(point[0] + offset_y + 1),
                int(point[1] + offset_x),
            ) not in dedup_set:
                dedup_set.add((int(point[0] + offset_y + 1), int(point[1] + offset_x)))
                deduped_points.append(
                    (int(point[0] + offset_y + 1), int(point[1] + offset_x), color)
                )
            elif (
                int(point[0] + offset_y - 1),
                int(point[1] + offset_x),
            ) not in dedup_set:
                dedup_set.add((int(point[0] + offset_y - 1), int(point[1] + offset_x)))
                deduped_points.append(
                    (int(point[0] + offset_y - 1), int(point[1] + offset_x), color)
                )
            elif (
                int(point[0] + offset_y),
                int(point[1] + offset_x + 1),
            ) not in dedup_set:
                dedup_set.add((int(point[0] + offset_y), int(point[1] + offset_x + 1)))
                deduped_points.append(
                    (int(point[0] + offset_y), int(point[1] + offset_x + 1), color)
                )
            elif (
                int(point[0] + offset_y),
                int(point[1] + offset_x - 1),
            ) not in dedup_set:
                dedup_set.add((int(point[0] + offset_y), int(point[1] + offset_x - 1)))
                deduped_points.append(
                    (int(point[0] + offset_y), int(point[1] + offset_x - 1), color)
                )
            elif (
                int(point[0] + offset_y + 1),
                int(point[1] + offset_x + 1),
            ) not in dedup_set:
                dedup_set.add(
                    (int(point[0] + offset_y + 1), int(point[1] + offset_x + 1))
                )
                deduped_points.append(
                    (int(point[0] + offset_y + 1), int(point[1] + offset_x + 1), color)
                )
            elif (
                int(point[0] + offset_y - 1),
                int(point[1] + offset_x - 1),
            ) not in dedup_set:
                dedup_set.add(
                    (int(point[0] + offset_y - 1), int(point[1] + offset_x - 1))
                )
                deduped_points.append(
                    (int(point[0] + offset_y - 1), int(point[1] + offset_x - 1), color)
                )
            elif (
                int(point[0] + offset_y + 1),
                int(point[1] + offset_x - 1),
            ) not in dedup_set:
                dedup_set.add(
                    (int(point[0] + offset_y + 1), int(point[1] + offset_x - 1))
                )
                deduped_points.append(
                    (int(point[0] + offset_y + 1), int(point[1] + offset_x - 1), color)
                )
            elif (
                int(point[0] + offset_y - 1),
                int(point[1] + offset_x + 1),
            ) not in dedup_set:
                dedup_set.add(
                    (int(point[0] + offset_y - 1), int(point[1] + offset_x + 1))
                )
                deduped_points.append(
                    (int(point[0] + offset_y - 1), int(point[1] + offset_x + 1), color)
                )
            else:
                print("WARN: point not added to deduped_points, we tried our best")
                print(point)
    rotated_points = deduped_points

    # Place the points in the new board
    for pointy, pointx, color in rotated_points:
        new_board_array[pointy, pointx] = color

    # Return the new Board
    return Board(__root__=new_board_array.tolist())


def doubleBoard(
    boardIn: Board,
    separation: int,
    is_horizontal_cat: bool,
    z_index_of_original: bool = 0,
) -> Board:
    """
    This function takes a Board and returns a new Board with the input board doubled.
    If is_horizontal_cat is True, the board is horizontally concatenated otherwise concatenation is vertical.
    Separation is the number of pixels between the two boards.
    This function supports negative separation values. In this case, one board is cropped to accomodate the other.
    if separation is negative, the z_index_of_original is used to determine which board lays ontop of the other.
    """

    np_board = boardIn.np
    ret_board_size = [len(np_board), len(np_board[0])]
    if is_horizontal_cat:
        ret_board_size[1] = ret_board_size[1] * 2 + separation
    else:
        ret_board_size[0] = ret_board_size[0] * 2 + separation
    ret_board = np.zeros(ret_board_size)

    if is_horizontal_cat:
        if z_index_of_original == 0:
            ret_board[:, : len(np_board[0])] = np_board
            ret_board[:, -len(np_board[0]) :] = np_board
        else:
            ret_board[:, -len(np_board[0]) :] = np_board
            ret_board[:, : len(np_board[0])] = np_board

    else:
        if z_index_of_original == 0:
            ret_board[: len(np_board), :] = np_board
            ret_board[-len(np_board) :, :] = np_board
        else:
            ret_board[-len(np_board) :, :] = np_board
            ret_board[: len(np_board), :] = np_board

    return Board(__root__=ret_board.tolist())


def doubleInputBoard(
    board_pair: BoardPair,
    separation: int,
    is_horizontal_cat: bool,
    z_index_of_original: bool = 0,
) -> BoardPair:
    board_pair.input = doubleBoard(
        board_pair.input, separation, is_horizontal_cat, z_index_of_original
    )
    return board_pair


def rotate(board_pair: BoardPair, num_rotations: int = 0) -> BoardPair:
    """
    This function takes a Board and returns a new Board that is rotated 90 degrees `num_rotations` times
    """
    assert num_rotations >= 0 and num_rotations < 4
    if num_rotations == 0:
        return board_pair
    return BoardPair(
        input=Board(__root__=np.rot90(board_pair.input.np, num_rotations).tolist()),
        output=Board(__root__=np.rot90(board_pair.output.np, num_rotations).tolist()),
    )


def reflect(
    board_pair: BoardPair, x_axis: bool = False, y_axis: bool = False
) -> BoardPair:
    """
    This function takes a Board and returns a new Board that is reflected over one of the following: x-axis,y-axis, xy-axes, no axes
    """

    if not x_axis and not y_axis:
        return board_pair.copy()

    if x_axis and y_axis:
        flip = None
    elif x_axis:
        flip = 0
    elif y_axis:
        flip = 1
    if flip is not None:
        return BoardPair(
            input=Board(__root__=np.flip(board_pair.input.np, flip).tolist()),
            output=Board(__root__=np.flip(board_pair.output.np, flip).tolist()),
        )
    else:
        # np.transpose
        return BoardPair(
            input=Board(__root__=np.transpose(board_pair.input.np).tolist()),
            output=Board(__root__=np.transpose(board_pair.output.np).tolist()),
        )


def padInputOutput(board: BoardPair, size: int, pad_value: int) -> BoardPair:
    input_np_board = board.input.np
    output_np_board = board.output.np
    input_np_board = padBoardHelper(input_np_board, size, pad_value)
    output_np_board = padBoardHelper(output_np_board, size, pad_value)
    return BoardPair(input=input_np_board.tolist(), output=output_np_board.tolist())


def padInputOnly(board: BoardPair, size: int, pad_value: int) -> BoardPair:
    input_np_board = board.input.np
    input_np_board = padBoardHelper(input_np_board, size, pad_value)
    return BoardPair(input=input_np_board.tolist(), output=board.output)


def padBoardHelper(boardIn: np.ndarray, size: int, pad_value: int) -> np.ndarray:
    boardIn = np.pad(boardIn, ((size, size), (size, size)), constant_values=pad_value)
    return boardIn


def padBoardHorizontal(boardIn: Board, size: int, pad_value: int) -> Board:
    boardIn = boardIn.np
    boardIn = np.pad(boardIn, ((0, 0), (size, 0)), constant_values=pad_value)
    return Board(__root__=boardIn.tolist())


def padBoardVertical(boardIn: Board, size: int, pad_value: int) -> Board:
    boardIn = boardIn.np
    boardIn = np.pad(boardIn, pad_width=((size, 0), (0, 0)), constant_values=pad_value)
    return Board(__root__=boardIn.tolist())


def superResolution(bp: BoardPair, factor: int, stretch_axis: Direction) -> BoardPair:
    bp.input = superResolutionBoard(bp.input, factor, stretch_axis)
    bp.output = superResolutionBoard(bp.output, factor, stretch_axis)
    return bp


def superResolutionBoard(boardIn: Board, factor: int, stretch_axis: Direction) -> Board:
    """
    This function takes a Board and returns a new Board that is super-resolved.
    The super-resolution is done by stretching the board in the specified direction.
    """
    np_board = boardIn.np
    if stretch_axis == Direction.horizontal:
        np_board = np.repeat(np_board, factor, axis=1)
    elif stretch_axis == Direction.vertical:
        np_board = np.repeat(np_board, factor, axis=0)
    elif stretch_axis == Direction.both:
        np_board = np.repeat(np_board, factor, axis=0)
        np_board = np.repeat(np_board, factor, axis=1)
    return Board(__root__=np_board.tolist())


def taurusTranslate(
    boardPair: BoardPair, x_to_rem_choice, y_to_rem_choice, dir_choice_x, dir_choice_y
) -> BoardPair:
    input_board = taurusTranslateBoard(
        boardPair.input, x_to_rem_choice, y_to_rem_choice, dir_choice_x, dir_choice_y
    )
    output_board = taurusTranslateBoard(
        boardPair.output, x_to_rem_choice, y_to_rem_choice, dir_choice_x, dir_choice_y
    )
    return BoardPair(input=input_board, output=output_board)


def taurusTranslateBoard(
    boardIn: Board, x_translate, x_dir, y_translate, y_dir
) -> Board:
    """
    This function takes a Board and returns a new Board that is translated by the specified amount.
    The translation is done by shifting the board in the specified direction.
    """
    np_board = boardIn.np
    x_translate = x_translate * x_dir
    y_translate = y_translate * y_dir
    np_board = np.roll(np_board, x_translate, axis=1)
    np_board = np.roll(np_board, y_translate, axis=0)
    return Board(__root__=np_board.tolist())


def quasiRotate(
    board_pair: BoardPair, angleHor, angleVer, startTopHor, startLeftVer, do_hor_first
) -> BoardPair:
    board_pair.input = quasiRotateBothAxes(
        board_pair.input, angleHor, angleVer, do_hor_first, startTopHor, startLeftVer
    )
    board_pair.output = quasiRotateBothAxes(
        board_pair.output, angleHor, angleVer, do_hor_first, startTopHor, startLeftVer
    )
    return board_pair


def quasiRotateBoardHor(boardIn: Board, angle, startTopHor) -> Board:
    def _helper(boardIn, angle):
        assert angle >= 0 and angle <= 45
        y_component = np.tan(np.deg2rad(angle)) * np.arange(len(boardIn.np))
        y_component = np.round(y_component)
        translated_point_list = []
        new_width = len(boardIn.np[0]) + int(np.max(y_component))
        for y in range(len(boardIn.np)):
            for x in range(len(boardIn.np[0])):
                if boardIn.np[y][x] != 0:
                    new_x = int(x + y_component[y])
                    translated_point_list.append(((y, new_x), boardIn.np[y][x]))
        new_height = len(boardIn.np)
        ret_board = np.zeros((new_height, new_width))
        for point, color in translated_point_list:
            ret_board[point[0], point[1]] = color
        return Board(__root__=ret_board.tolist(), dtype=boardIn.np.dtype)

    if not startTopHor:
        boardIn = Board(__root__=np.flip(boardIn.np, 0).tolist())
    ret_board = _helper(boardIn, angle)
    if not startTopHor:
        ret_board = Board(__root__=np.flip(ret_board.np, 0).tolist())
    return ret_board


def quasiRotateBoardVer(BoardIn: Board, angle, startLeftVer) -> Board:
    def _helper(BoardIn: Board, angle):
        flip_x_y = np.transpose(BoardIn.np)
        ret_board = quasiRotateBoardHor(Board(__root__=flip_x_y.tolist()), angle, True)
        ret_board = np.transpose(ret_board.np)
        return Board(__root__=ret_board.tolist(), dtype=BoardIn.np.dtype)

    if not startLeftVer:
        BoardIn = Board(__root__=np.flip(BoardIn.np, 1).tolist())
    ret_board = _helper(BoardIn, angle)
    if not startLeftVer:
        ret_board = Board(__root__=np.flip(ret_board.np, 1).tolist())
    return ret_board


def quasiRotateBothAxes(
    boardIn: Board,
    angleHor,
    angleVer,
    do_hor_first=True,
    startTopHor=True,
    startLeftVer=True,
) -> Board:
    if do_hor_first:
        ret_board = quasiRotateBoardHor(boardIn, angleHor, startTopHor)
        ret_board = quasiRotateBoardVer(ret_board, angleVer, startLeftVer)
    else:
        ret_board = quasiRotateBoardVer(boardIn, angleVer, startLeftVer)
        ret_board = quasiRotateBoardHor(ret_board, angleHor, startTopHor)
    return ret_board
