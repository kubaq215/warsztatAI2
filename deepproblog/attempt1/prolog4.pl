%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prolog2.pl for Sudoku demo %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(lists)).  % For member/2, append/3, etc.
:- use_module(library(apply)).


% 1) Neural network predicate for images: digit/2
%    Ties to your DeepProbLog net named "mnist_net" with possible digits 1..9.
nn(mnist_net,[ImgTensor],Digit,[1,2,3,4,5,6,7,8,9]) :: digit(ImgTensor,Digit).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Interpreting each puzzle cell as a digit 
%    - If puzzle cell is an integer => that cell is already filled
%    - If puzzle cell is tensor(...) => call the NN to classify
%    - If puzzle cell is none => remain a free variable (logic fills it)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cell_digit(+CellTerm, -Digit)
%   This predicate unifies `Digit` with the actual integer for that cell.
%   `CellTerm` is how you represent the cell in your puzzle_solve/1 query:
%
%     - integer(N)        => means the puzzle already has digit N in that cell
%     - tensor(...)       => means the puzzle has an image to classify
%     - none              => means the puzzle cell is empty / unknown
%
cell_digit(none, D) :- 
    % Let D be a free variable in the standard Prolog sense, 
    % but we must restrict it to 1..9 eventually. We'll do that 
    % in "valid_unit" or "all_diff" checks. 
    % If you want to explicitly force 1..9, do: member(D,[1,2,3,4,5,6,7,8,9]).
    member(D, [1,2,3,4,5,6,7,8,9]).  % standard approach to fill the cell

cell_digit(X, X) :- 
    integer(X), 
    X >= 1, X =< 9.     % a pre-filled digit from 1..9

cell_digit(tensor(Source,Idx,Row,Col), D) :-
    % We invoke the NN to classify the image in [1..9].
    digit(tensor(Source,Idx,Row,Col), D). 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) Each row/column/box must have distinct digits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% valid_unit(ListOfCells)
%   Succeeds if the cells in this row/column/box all unify to 
%   distinct digits 1..9.
valid_unit(Cells) :-
    maplist(cell_digit, Cells, Digits),  % unify each cell with a digit
    all_different(Digits).              % no duplicates

% all_different(+Digits) => unify them all as distinct
all_different([]).
all_different([X|Xs]) :-
    not_in(X, Xs),
    all_different(Xs).

not_in(_, []).
not_in(X, [Y|Ys]) :-
    X \= Y,
    not_in(X, Ys).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4) puzzle_solve(Puzzle): 
%    The Sudoku constraints:
%      - 9 rows, each row is valid_unit
%      - 9 columns, each column is valid_unit
%      - 3×3 boxes, each is valid_unit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

puzzle_solve(Puzzle) :-
    length(Puzzle, 9),          % Must have 9 rows
    maplist(check_length_9, Puzzle),

    % Rows
    maplist(valid_unit, Puzzle),

    % Columns
    transpose(Puzzle, Cols),
    maplist(valid_unit, Cols),

    % 3×3 Boxes
    boxes_3x3(Puzzle, Boxes),
    maplist(valid_unit, Boxes).

check_length_9(Row) :- length(Row,9).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5) Transpose a 9×9 matrix in Prolog
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

transpose([],[]).
transpose([R|Rs], Ts) :-
    transpose_(R, [R|Rs], Ts).

transpose_([],_,[]).
transpose_([_|Xs], Rows, [Col|Cols]) :-
    first_column(Rows, Col, RestRows),
    transpose_(Xs, RestRows, Cols).

first_column([], [], []).
first_column([[H|T]|Rs], [H|Hs], [T|Ts]) :-
    first_column(Rs, Hs, Ts).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6) 3×3 box extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boxes_3x3(Rows, BoxList) :-
    Rows = [R0,R1,R2,R3,R4,R5,R6,R7,R8],
    three_box_rows(R0,R1,R2, Boxes1),
    three_box_rows(R3,R4,R5, Boxes2),
    three_box_rows(R6,R7,R8, Boxes3),
    append(Boxes1, Boxes2, Temp),
    append(Temp, Boxes3, BoxList).

three_box_rows(RA,RB,RC, [Box1,Box2,Box3]) :-
    row_3cols(RA,A1,A2,A3),
    row_3cols(RB,B1,B2,B3),
    row_3cols(RC,C1,C2,C3),
    append(A1,B1,AB1), append(AB1,C1,Box1),
    append(A2,B2,AB2), append(AB2,C2,Box2),
    append(A3,B3,AB3), append(AB3,C3,Box3).

row_3cols(Row, Part1, Part2, Part3) :-
    Row = [C0,C1,C2, C3,C4,C5, C6,C7,C8],
    Part1 = [C0,C1,C2],
    Part2 = [C3,C4,C5],
    Part3 = [C6,C7,C8].
