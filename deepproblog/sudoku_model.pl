% Define the neural network for recognizing digits in cells
nn(sudoku_net, [X], Y, [1,2,3,4,5,6,7,8,9]) :: cell(X, Y).

% Sudoku predicate that applies row, column, and sub-grid constraints
sudoku(Board) :-
    valid_rows(Board),
    valid_columns(Board),
    valid_squares(Board).

% Ensure each row in the Board has all unique values (1-9)
valid_rows(Board) :-
    maplist(all_different, Board).

% Ensure each column in the Board has all unique values (1-9)
valid_columns(Board) :-
    transpose(Board, Transposed),
    maplist(all_different, Transposed).

% Ensure each 3x3 sub-grid in the Board has all unique values (1-9)
valid_squares(Board) :-
    squares(Board, Squares),
    maplist(all_different, Squares).

% Helper predicate: all_different(List) is true if all elements in List are distinct
all_different([]).
all_different([H|T]) :-
    \+ member(H, T),
    all_different(T).

% Helper predicate: transpose(Board, Transposed) transposes a 9x9 Board matrix
transpose([[]|_], []).
transpose(Matrix, [Row|Rows]) :-
    maplist(list_head_tail, Matrix, Row, Tails),
    transpose(Tails, Rows).

list_head_tail([H|T], H, T).

% Helper predicate: squares(Board, Squares) extracts all 3x3 sub-grids from Board
squares(Board, Squares) :-
    Board = [R1, R2, R3, R4, R5, R6, R7, R8, R9],
    three_by_three(R1, R2, R3, S1, S2, S3),
    three_by_three(R4, R5, R6, S4, S5, S6),
    three_by_three(R7, R8, R9, S7, S8, S9),
    Squares = [S1, S2, S3, S4, S5, S6, S7, S8, S9].

% Helper predicate: three_by_three(R1, R2, R3, S1, S2, S3) extracts 3x3 blocks from three rows
three_by_three([], [], [], [], [], []).
three_by_three([A,B,C|R1], [D,E,F|R2], [G,H,I|R3],
               [A,B,C|S1], [D,E,F|S2], [G,H,I|S3]) :-
    three_by_three(R1, R2, R3, S1, S2, S3).
