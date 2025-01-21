:- use_module(library(lists)).  % For member/2, append/3, etc.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Neural Network Interface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If the cell is an actual image (tensor(...)), the neural network chooses
% one of [1..9]. We call that predicate 'digit(ImageTerm, Digit)'.
nn(mnist_net, [ImageTerm], Digit, [1,2,3,4,5,6,7,8,9]) :: digit_nn(ImageTerm, Digit).

% For none cells, we just allow 1..9 by backtracking.
digit(none, D) :- 
    digit_domain(D).

% For real images, use the NN:
digit(ImageTerm, D) :-
    ImageTerm \= none,
    digit_nn(ImageTerm, D).

% 1..9 domain
digit_domain(1). digit_domain(2). digit_domain(3).
digit_domain(4). digit_domain(5). digit_domain(6).
digit_domain(7). digit_domain(8). digit_domain(9).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sudoku Logic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
distinct([]).
distinct([X|Xs]) :-
    \+ member(X, Xs),
    distinct(Xs).

sudoku(Rows) :-
    length(Rows, 9),
    maplist(check_length_9, Rows),
    maplist(distinct, Rows),
    transpose(Rows, Columns),
    maplist(distinct, Columns),
    blocks_ok(Rows).

check_length_9(Row) :- length(Row, 9).

blocks_ok([A,B,C,D,E,F,G,H,I]) :-
    blocks3(A,B,C),
    blocks3(D,E,F),
    blocks3(G,H,I).

blocks3([], [], []).
blocks3([X1,X2,X3|R1], [X4,X5,X6|R2], [X7,X8,X9|R3]) :-
    distinct([X1,X2,X3,X4,X5,X6,X7,X8,X9]),
    blocks3(R1, R2, R3).

transpose([], []).
transpose([Row|Rows], T) :-
    transpose_row(Row, [Row|Rows], T).

transpose_row([], _, []).
transpose_row([_|_], Matrix, [Col|Cols]) :-
    first_column(Matrix, Col, Rest),
    transpose_row([], Rest, Cols).

first_column([], [], []).
first_column([[X|Xs]|Rows], [X|Col], [Xs|Rest]) :-
    first_column(Rows, Col, Rest).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% puzzle_solve/1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each cell in a 9x9 puzzle, call digit/2. Then check Sudoku constraints.
puzzle_solve(ImageRows) :-
    sudoku_image_list(ImageRows, ValueRows),
    sudoku(ValueRows).

sudoku_image_list([], []).
sudoku_image_list([ImgRow|ImgRows], [ValRow|ValRows]) :-
    maplist(digit, ImgRow, ValRow),
    sudoku_image_list(ImgRows, ValRows).
