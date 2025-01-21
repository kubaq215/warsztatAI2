:- use_module(library(lists)).  % For member/2, append/3, etc.
:- use_module(library(apply)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Neural Network + Wrappers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We define "digit_nn/2" as the actual hook to the neural net.
% ProbLog's "nn(...)" annotation says: if the first argument is an image (not none),
% then 'digit_nn(Image, D)' is proven by the neural network in domain [1..9].
%
nn(mnist_net, [Img], D, [1,2,3,4,5,6,7,8,9]) :: digit_nn(Img, D).

% For cells that are none, we just pick any digit 1..9:
digit_none(D) :-
    digit_domain(D).

digit_domain(1).  digit_domain(2).  digit_domain(3).
digit_domain(4).  digit_domain(5).  digit_domain(6).
digit_domain(7).  digit_domain(8).  digit_domain(9).

% Wrapper "digit/2" that adds a debug message *every time* it's called:
% digit(none, D) :-
%     debugprint(1, "digit/2 => none => enumerating 1..9", []),
%     digit_none(D),
%     debugprint(1, "digit/2 => none => D=~w", [D]).

% digit(X, D) :-
%     X \= none,
%     debugprint(1, "digit/2 => calling NN for X=~w", [X]),
%     digit_nn(X, D),
%     debugprint(1, "digit/2 => NN recognized => D=~w", [D]).

digit(X, D) :-
    (
        ( X = none,
          debugprint(1, "digit/2 => none => enumerating 1..9", []),
          digit_none(D),
          debugprint(1, "digit/2 => none => D=~w", [D])
        )
    ;
        ( X \= none,
          debugprint(1, "digit/2 => calling NN for X=~w", [X]),
          digit_nn(X, D),
          debugprint(1, "digit/2 => NN recognized => D=~w", [D])
        )
    ),
    debugprint(1, "digit/2 => Final digit value D=~w", [D]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Sudoku Logic (distinct, transpose, blocks)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Basic distinct/1 check with debug for demonstration
distinct([]) :-
    debugprint(1, "distinct/1: Reached base case (empty list).", []).
distinct([X|Xs]) :-
    debugprint(1, "distinct/1: Checking element ~w among ~w", [X, Xs]),
    \+ member(X, Xs),
    distinct(Xs),
    debugprint(1, "distinct/1: Element ~w is distinct in its list.", [X]).

sudoku(Rows) :-
    debugprint(1, "sudoku/1: Called with Rows=~w", [Rows]),
    length(Rows, 9),
    maplist(check_length_9, Rows),
    maplist(distinct, Rows),      % Rows must have distinct digits
    transpose(Rows, Columns),
    maplist(distinct, Columns),   % Columns must have distinct digits
    blocks_ok(Rows),
    debugprint(1, "sudoku/1: Puzzle passed distinctness checks", []).

check_length_9(Row) :-
    length(Row, 9),
    debugprint(1, "check_length_9: Row ~w has length 9", [Row]).

blocks_ok([A,B,C,D,E,F,G,H,I]) :-
    debugprint(1, "blocks_ok/1: Checking block groups [A,B,C], [D,E,F], [G,H,I]", []),
    blocks3(A,B,C),
    blocks3(D,E,F),
    blocks3(G,H,I).

blocks3([], [], []) :-
    debugprint(1, "blocks3/3: Base case (no more sub-block cells).", []).
blocks3([X1,X2,X3|R1], [X4,X5,X6|R2], [X7,X8,X9|R3]) :-
    debugprint(1, "blocks3/3: Checking sub-block ~w", [[X1,X2,X3],[X4,X5,X6],[X7,X8,X9]]),
    distinct([X1,X2,X3, X4,X5,X6, X7,X8,X9]),
    blocks3(R1, R2, R3).

% Simple transpose if no built-in is available.
transpose([], []) :-
    debugprint(1, "transpose/2: Base case (empty list).", []).

transpose([[]|_], []) :-
    debugprint(1, "transpose/2: Base case (empty row).", []).

transpose(Matrix, [Column|Columns]) :-
    debugprint(1, "transpose/2: Matrix before transposing ~w", [Matrix]),
    maplist(first_rest_debug, Matrix, Column, Remainder),
    debugprint(1, "transpose/2: Extracted column ~w, Remaining matrix ~w", [Column, Remainder]),
    transpose(Remainder, Columns).

% Helper predicate that takes the head and tail of each list
first_rest([H|T], H, T) :-
    debugprint(1, "first_rest/3: Extracting head ~w, tail ~w", [H, T]).

% Debug version of first_rest
first_rest_debug(Row, Head, Tail) :-
    first_rest(Row, Head, Tail),
    debugprint(1, "first_rest_debug/3: Row ~w => Head ~w, Tail ~w", [Row, Head, Tail]).

first_column([], [], []) :-
    debugprint(1, "first_column/3: No more rows.", []).
first_column([[X|Xs]|Rows], [X|Col], [Xs|Rest]) :-
    debugprint(1, "first_column/3: Extracting ~w", [X]),
    first_column(Rows, Col, Rest).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. puzzle_solve/1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% puzzle_solve/1: For each cell (image or none), call digit/2 => get digits,
% then solve with sudoku/1.

puzzle_solve(ImageRows) :-
    debugprint(1, "puzzle_solve/1: Called with ImageRows=~w", [ImageRows]),
    sudoku_image_list(ImageRows, ValueRows),
    debugprint(1, "puzzle_solve/1: ValueRows=~w", [ValueRows]),
    sudoku(ValueRows),
    debugprint(1, "puzzle_solve/1: Successfully solved puzzle", []).

sudoku_image_list([], []) :-
    debugprint(1, "sudoku_image_list/2: Base case, no more rows.", []).
    
sudoku_image_list([ImgRow|ImgRows], [ValRow|ValRows]) :-
    debugprint(1, "sudoku_image_list/2: Mapping ImgRow=~w", [ImgRow]),
    maplist(debug_digit_map, ImgRow, ValRow),
    debugprint(1, "sudoku_image_list/2: Mapped => ValRow=~w", [ValRow]),
    sudoku_image_list(ImgRows, ValRows).

debug_digit_map(Cell, Digit) :-
    debugprint(1, "debug_digit_map: Processing Cell=~w", [Cell]),
    digit(Cell, Digit),
    debugprint(1, "debug_digit_map: Cell=~w mapped to Digit=~w", [Cell, Digit]).