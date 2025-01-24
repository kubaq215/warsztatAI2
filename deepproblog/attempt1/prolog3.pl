:- use_module(library(lists)).  % For member/2, append/3, etc.
:- use_module(library(apply)).
:- use_module(library(cut)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Neural Network + Wrappers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We define "digit_nn/2" as the hook to the neural net. ProbLog's "nn(...)" 
% annotation says: if the first argument is an image (not none),
% then 'digit_nn(Image, D)' is derived from the neural network in domain [1..9].
%
nn(mnist_net, [Img], D, [1,2,3,4,5,6,7,8,9]) :: digit_nn(Img, D).


digit_none(_).

% Wrapper "digit/2" that adds debugging messages.
digit(X, D, Res) :-
    (
        ( X = none,
          digit_none(D)
        )
    ;
        ( X \= none,
          digit_nn(X, D),
          Res is D
        )
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove an item from a list %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) Define indexed rules mirroring each clause:
r_ri(1, removeItem(Item, List, Out)) :-
    select(Item, List, Out).

r_ri(2, removeItem(_, List, List)).

% 2) Wrap removeItem/3 with cut:
removeItem(A, B, C) :-
    cut(r_ri(removeItem(A, B, C))).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All different values in a list %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inequal(X, Y) :-
    % debugprint(1, "inequal/2: Called with X=~w, Y=~w", [X, Y]),
    (
      ( var(X), var(Y),
        % debugprint(1, "inequal/2: Both arguments are `_`, ignoring check => success", []),
        true
      )
    ;   % Case 2: Both arguments are something else => do the usual difference check
        X \== Y
    ).

allDifferent([]). %:- 
    % debugprint(1, "allDifferent/1: Base case (empty list).", []).
allDifferent([X|Xss]) :- 
    % debugprint(1, "allDifferent/1: Called with X=~w, Xss=~w", [X, Xss]),
    maplist(inequal(X), Xss), 
    allDifferent(Xss).

checkConsistency(Problem) :-
    % debugprint(1, "chckConsist/1:  Called with Problem=~w", [Problem]),
    maplist(allDifferent, Problem).
    % debugprint(1, "chckConsist/1:  Problem is consistent", []).

%%%%%%%%%%%%%%%%%%%%%%%%
% Transpose the matrix %
%%%%%%%%%%%%%%%%%%%%%%%%

transpose(Ls, Ts) :-
    debugprint(1, "transpose/2: Called with Ls=~w", [Ls]),
    lists_transpose(Ls, Ts),
    debugprint(1, "transpose/2: Transposed ~w", [Ts]).

lists_transpose([], []).% :-
    % debugprint(1, "lists_transpose/2: Base case (empty list).", []).
lists_transpose([L|Ls], Ts) :-
    % debugprint(1, "lists_transpose/2: Called with L=~w", [L]),
    foldl(transpose_, L, Ts, [L|Ls], _).

transpose_(_, Fs, Lists0, Lists) :-
    % debugprint(1, "transpose_/4: Called with Fs=~w, Lists0=~w", [Fs, Lists0]),
    maplist(list_first_rest, Lists0, Fs, Lists).

list_first_rest([L|Ls], L, Ls).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reduce domain of a list   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % 1) Define indexed rules for each original clause:
% r_rd(1, reduceDomain(_, [], [])). %:-
%     % debugprint(1, "reduceDomain/3: Base case (empty domain).", []).
% r_rd(2, reduceDomain([], Domain, Domain)). %:-
%     % debugprint(1, "reduceDomain/3: Base case (empty list).", []).
% r_rd(3, reduceDomain([H|T], Domain, Out)) :-
%     nonvar(H),
%     % debugprint(1, "reduceDomain/3: Case with real number", [H]),
%     % debugprint(1, "reduceDomain/3: Called with Number=~w, Rest=~w, Domain=~w", [H, T, Domain]),
%     removeItem(H, Domain, DomainWithoutValue),
%     reduceDomain(T, DomainWithoutValue, Out).
% r_rd(4, reduceDomain([H|T], Domain, Out)) :-
%     var(H),
%     % debugprint(1, "reduceDomain/3: Case with anonymous variable", [H]),
%     % debugprint(1, "reduceDomain/3: Called with Var=~w, Rest=~w, Domain=~w", [H, T, Domain]),
%     reduceDomain(T, Domain, Out).

% % 2) Provide a wrapper that calls the cut-libray:
% reduceDomain(Vars, Domain, ReducedDomain) :-
%     % debugprint(1, "reduceDomain/3: Called with Vars=~w, Domain=~w", [Vars, Domain]),
%     cut(r_rd(reduceDomain(Vars, Domain, ReducedDomain))).

reduceDomain(_,[],[]).
reduceDomain([],Domain,Domain).

reduceDomain([H|T],Domain,Out) :- 
    nonvar(H), removeItem(H,Domain,DomainWithoutValue), reduceDomain(T,DomainWithoutValue,Out).

reduceDomain([H|T],Domain,Out) :- 
    var(H), reduceDomain(T,Domain,Out).

%%%%%%%%%%%%%%%%%
% Get 3x3 boxes %
%%%%%%%%%%%%%%%%%

% 1) Define indexed rules for each of the three clauses:
r(1, getBoxStack([], _, _, Buffer, [Buffer], [])). %:-
    % debugprint(1, "r_getBoxStack/7: Base case (empty list).", []).
r(2, getBoxStack(Sudoku, Width, Height, Buffer, [Buffer|RestOfBoxes], RemainingRows)) :-
    % debugprint(1, "r_getBoxStack/7: Called with Sudoku=~w, Width=~w, Height=~w, Buffer=~w", [Sudoku, Width, Height, Buffer]),
    SizeOfBox is Width * Height,
    length(Buffer, SizeOfBox),
    getBoxStack(Sudoku, Width, Height, [], RestOfBoxes, RemainingRows).
r(3, getBoxStack([CurrentRow|Frontier], Width, Height, OldBuffer, Boxes, [RestOfCurrentRow|RemainingRows])) :-
    % debugprint(1, "r_getBoxStack/7: Called with Frontier=~w, Width=~w, Height=~w, OldBuffer=~w", [Frontier, Width, Height, OldBuffer]),
    length(CurrentStackSlice, Width),
    append(CurrentStackSlice, RestOfCurrentRow, CurrentRow),
    append(CurrentStackSlice, OldBuffer, NewBuffer),
    getBoxStack(Frontier, Width, Height, NewBuffer, Boxes, RemainingRows).

% 2) Wrap the call to getBoxStack/7 with 'cut/1'
getBoxStack(Sudoku, Width, Height, Buffer, Boxes, RemainingRows) :-
    % debugprint(1, "getBoxStack/6: Called with Sudoku=~w, Width=~w, Height=~w, Buffer=~w", [Sudoku, Width, Height, Buffer]),
    cut(r(getBoxStack(Sudoku, Width, Height, Buffer, Boxes, RemainingRows))).

% 2. Define each 'getBoxes_' clause as indexed rules (r/5 here)
r(1, getBoxes_([], _, _, [])).
r(2, getBoxes_([[]|_], _, _, [])).
r(3, getBoxes_(Sudoku, Width, Height, Out)) :-
    % debugprint(1, "r_getBoxes_/4: Called with Sudoku=~w, W=~w, H=~w", [Sudoku, W, H]),
    getBoxStack(Sudoku, Width, Height, [], Stack, Rest),
    append(Stack, OtherBoxes, Out),
    % Recurse with the same wrapper:
    getBoxes_(Rest, Width, Height, OtherBoxes).

% 3. Provide a wrapper for getBoxes_/4
getBoxes_(Sudoku, Width, Height, Out) :-
    % debugprint(1, "getBoxes_/4: Called with S=~w, W=~w, H=~w", [S, W, H]),
    cut(r(getBoxes_(Sudoku, Width, Height, Out))).

getBoxes([H|T], BoxWidth, BoxHeight, Out) :-
        debugprint(1, "getBoxes/4: Called with H=~w, T=~w, BoxWidth=~w, BoxHeight=~w", [H, T, BoxWidth, BoxHeight]),
        length([H|T], MatrixSize), 
        length(H, MatrixSize), 
        Y is (MatrixSize mod BoxHeight),
        X is (MatrixSize mod BoxWidth),
        X = Y, Y = 0, BoxWidth \= 0, BoxHeight \= 0,
        MatrixSize is BoxWidth * BoxHeight,
        getBoxes_([H|T], BoxWidth, BoxHeight, Out).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contatenate rows, cols and boxes %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

getProblem(H, W, Rows,Out) :-
    debugprint(1, "getProblem/3: Called with H=~w, W=~w, Rows=~w", [H, W, Rows]),
    transpose(Rows,Cols), 
    getBoxes(Rows,W,H,Boxes), 
    append(Rows,Cols,Temp),
    append(Temp,Boxes,Out).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Try to solve the problem  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% r_sp(1, solvePiece([], _, _, _)) :-    % Clause 1
%     % debugprint(1, "solvePiece/4:   Base case (empty list).", []),
%     true.                              % (the cut was here in plain Prolog)

% r_sp(2, solvePiece([Item|Rest], Dom, List, Problem)) :-   % Clause 2
%     % debugprint(1, "solvePiece/4:   Called with Item=~w, Rest=~w, Dom=~w, List=~w", [Item, Rest, Dom, List]),
%     nonvar(Item),
%     solvePiece(Rest, Dom, List, Problem).

% r_sp(3, solvePiece([CurrentVar|Vars], Domain, List, Problem)) :-  % Clause 3
%     % debugprint(1, "solvePiece/4:   Called with Vars=~w, Domain=~w, List=~w", [Vars, Domain, List]),
%     select(CurrentTip, Domain, NewDomain), % End of recursion
%     CurrentVar = CurrentTip,
%     checkConsistency(Problem),
%     solvePiece(Vars, NewDomain, List, Problem).

% solvePiece(Vars, ReducedDomain, List, Problem) :-
%     % debugprint(1, "solvePiece/4:   Called with Vars=~w, ReducedDomain=~w, List=~w", [Vars, ReducedDomain, List]),
%     cut(r_sp(solvePiece(Vars, ReducedDomain, List, Problem))).

% r_sp_3(1, solvePiece(Vars, Domain, Problem)) :-  % Clause 4
%     % debugprint(1, "solvePiece/3:   Called with Vars=~w, Domain=~w", [Vars, Domain]),
%     reduceDomain(Vars, Domain, ReducedDomain),
%     debugprint(1, "solvePiece/3: Vars=~w", [Vars]),
%     solvePiece(Vars, ReducedDomain, Vars, Problem).

% solvePiece(Row, Domain, Problem) :-
%     % debugprint(1, "solvePiece/3:   Called with Row=~w, Domain=~w, Problem=~w", [Row, Domain, Problem]),
%     cut(r_sp_3(solvePiece(Row, Domain, Problem))).

solvePiece([], _, _, _).

solvePiece([Item|Rest], Dom, List, Problem) :- 
    nonvar(Item), solvePiece(Rest, Dom, List, Problem).

solvePiece([CurrentVar|Vars], Domain, List, Problem) :- 
        select(CurrentTip, Domain, NewDomain),   % predicate select/3 fails if Domain is empty.
        CurrentVar = CurrentTip,
        checkConsistency(Problem),
        solvePiece(Vars, NewDomain, List, Problem).

solvePiece(Vars,Domain, Problem) :-
    reduceDomain(Vars,Domain,ReducedDomain), 
    solvePiece(Vars, ReducedDomain, Vars, Problem).

getSolution([],_) :- 
    debugprint(1, "getSolution/2: Sudoku solved, but the garbage persists.", []),
    true.
getSolution([H|T],Domain) :- 
    % debugprint(1, "getSolution/2: Called with H=~w, T=~w, Domain=~w", [H, T, Domain]),
    solvePiece(H,Domain,[H|T]), 
    getSolution(T,Domain).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    "Public" Predicates    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solveSudoku(H,W,Sudoku) :-
    debugprint(1, "solveSudoku/3: Called with H=~w, W=~w, Sudoku=~w", [H, W, Sudoku]),
    getProblem(H,W,Sudoku,Problem), 
    Nums is H*W, 
    numlist(1,Nums,Domain),
    % debugprint(1, "solveSudoku/3: Domain=~w", [Domain]),
    getSolution(Problem,Domain),
    % debugprint(1, "printing problem", []),
    % printSudoku(Problem),
    debugprint(1, "printing solution", []),
    printSudoku(Sudoku).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. puzzle_solve/1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% puzzle_solve/1: 
%  1) Convert each 9x9 'ImageRow' into a 'ValueRow' of digits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

puzzle_solve(ImageRows) :-
    debugprint(1, "puzzle_solve/1: Called with ImageRows=~w", [ImageRows]),
    puzzle_image_rows(ImageRows, ValueRows),
    debugprint(1, "puzzle_solve/1: ValueRows=~w", [ValueRows]),
    % printSudoku(ValueRows),
    solveSudoku(3,3,ValueRows),
    debugprint(1, "puzzle_solve/1: Successfully solved puzzle", []).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% puzzle_image_rows/2 : builds each row 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
puzzle_image_rows([], []) :-
    debugprint(1, "puzzle_image_rows/2: Base case, no more rows.", []).

puzzle_image_rows([ImgRow|ImgRows], [ValRow|ValRows]) :-
    row_image_to_digits(ImgRow, ValRow),   
    debugprint(1, "puzzle_image_rows/2: Built row => ~w", [ValRow]),
    puzzle_image_rows(ImgRows, ValRows).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% row_image_to_digits/3: Fill a single row's digits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
row_image_to_digits([], []).

row_image_to_digits([Cell|Cells], [Digit|Digits]) :-
    ( 
        ( Cell = none,
          % maybe not necessary, because we do the same in digit/2
          Digit = _
        )
    ;
        ( Cell \= none,
          % We call the digit(tensor(train(index, i, j)), Label) predicate created in notebook's SudokuPuzzleDataset.to_query()
          % and assign the result to Digit
          % when we do it with the label the nn gives us one digit, otherwise it splits the program into 9 different subtrees
          % which are then solved independently, so with this much calls it's not feasible
          % I don't think thats the way to do it to get the improved accuracy of the nn
          % to achieve that we would have to somehow make the label into the solution table 
          call(Cell, Digit) 
        )
    ),
    row_image_to_digits(Cells, Digits).

printSudoku([]).

printSudoku([FirstRow|Rest]) :-
    debugprint(1, "", [FirstRow]),
    printSudoku(Rest).