Now we add in functionality to detect dead/captured cells. There is a new class (Pattern) of which some instances have been added to the Position class. The function of this new class is to detect "patterns" of cells that produce dead or captured cells. It uses cubic coordinates to find these patterns. (See the first link in the further reading section for an explanation of cubic coordinates). A new command has been added to visualize which cells are captured or dead. The function win_move has been augmented to detect captured/dead cells. On each move, dead/captured cells are iteratively computed, and if the current player now has connected sides, the ptm-captured cells have to be added to the winset before it is returned. We also have to add an if statement to the beginning of the function to handle the case where the current player has won but there are no empty spaces on the board so the mustplay is empty. There is also two new functions, "dead" and "captured".

In its current state, this solver can handle up to 4x4 hex.

Further reading:
	https://www.redblobgames.com/grids/hexagons/
	http://webdocs.cs.ualberta.ca/~hayward/papers/revDom.pdf
	http://webdocs.cs.ualberta.ca/~hayward/355/hexnotesp1.pdf
