import random
import sys
import chess
import chess.engine
import chess.pgn
import joblib
import numpy as np
import pygame
import os
from RF_suggeset import capture_features, center_control_features,defense_features,piece_mobility, material_balance
#obtain directary for stockfish
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(__file__)
engine_path = os.path.join(base_path, "stockfish.exe")

pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 480 * 2 + 300, 480 * 2
board_width = 480 * 2
board_height = 480 * 2
comment_width = 300
comment_height = 300
SQUARE_SIZE = board_width // 8
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) #create a screen that is of this size
pygame.display.set_caption("Chess!")

# Loading images
Board_image = pygame.image.load('Images/Board.png')
board_image = pygame.transform.scale(Board_image, (board_width, board_height)) # transform the images to fits the screen size
piece_images = {
    'P': pygame.image.load('Images/white-pawn.png'),
    'N': pygame.image.load('Images/white-knight.png'),
    'B': pygame.image.load('Images/white-bishop.png'),
    'R': pygame.image.load('Images/White-rook.png'),
    'Q': pygame.image.load('Images/white-queen.png'),
    'K': pygame.image.load('Images/white-king.png'),
    'p': pygame.image.load('Images/black-pawn.png'),
    'n': pygame.image.load('Images/black-knight.png'),
    'b': pygame.image.load('Images/black-bishop.png'),
    'r': pygame.image.load('Images/black-rook.png'),
    'q': pygame.image.load('Images/black-queen.png'),
    'k': pygame.image.load('Images/black-king.png'),
}
for key in piece_images:
    piece_images[key] = pygame.transform.scale(piece_images[key], (SQUARE_SIZE, SQUARE_SIZE))

def print_board(screen, board, initial_square=None, dragging_piece=None, dragging_position=None):
    screen.blit(board_image, (0, 0))
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, row))
            if piece and (dragging_piece is None or chess.square(col, row) != initial_square):
                screen.blit(piece_images[piece.symbol()], (col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE))
    if dragging_piece and dragging_position:
        screen.blit(piece_images[dragging_piece.symbol()],
                    (dragging_position[0] - SQUARE_SIZE // 2, dragging_position[1] - SQUARE_SIZE // 2))

#conver square to pixel or vice versa
def square_to_pixel(square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE)
def pixel_to_square(x, y):
    horizontal = x // SQUARE_SIZE
    vertical = 7 - (y // SQUARE_SIZE)
    return chess.square(horizontal, vertical)

#grab the features of the board. Specifically, which piece is at which position of the board
def board_features(board):
    features = []

    features.append(int(board.turn))
    features.extend([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ])

    square_features = np.zeros(64, dtype=int)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece.piece_type
            if piece.color == chess.BLACK:
                value += 6
            square_features[square] = value
    features.extend(square_features.tolist())

    capture_feats = capture_features(board)
    features.extend(capture_feats.tolist())


    mobility_feats = piece_mobility(board)
    features.extend(mobility_feats)

    center_feats = center_control_features(board)
    features.extend(center_feats)

    defense_feats = defense_features(board)
    features.extend(defense_feats)

    white_value, black_value = material_balance(board)
    features.append(white_value)  # Total value of White's pieces
    features.append(black_value)
    features.append(board.fullmove_number)
    features.append(board.halfmove_clock)
    features.append(int(board.is_check()))
    features.append(int(board.is_checkmate()))
    features.append(int(board.is_stalemate()))

    return np.array(features)


def suggest_move(board, model, label_encoder):
    features = board_features(board).reshape(1, -1)

    move_encoded = model.predict(features)[0]
    move_uci = label_encoder.inverse_transform([move_encoded])[0]
    predicted_move = chess.Move.from_uci(move_uci)
    legal_moves = list(board.legal_moves)
    if predicted_move in legal_moves:
        return predicted_move
    else:
        return blow_it_up(board)
def blow_it_up(board):
    capturing_moves = [move for move in board.legal_moves if  board.is_capture(move)]
    print("generated illegal moves")
    if capturing_moves:
        return random.choice(capturing_moves)
    return random.choice(list(board.legal_moves))

def draw_buttons(screen):
    font = pygame.font.Font(None, 50)#create a font object
    replay_text = font.render("Replay", True, (255, 255, 255)) #apply the property of the font to text
    replay_rect = replay_text.get_rect(center=(board_width // 2 - 100, board_height // 2 + 50))#create a rectangular object at spcific posision
    replay_button = pygame.Rect(replay_rect.left - 20, replay_rect.top - 10, replay_rect.width + 40, replay_rect.height + 20) #adjest the rectangle

    quit_text = font.render("Quit", True, (255, 255, 255))
    quit_rect = quit_text.get_rect(center=(board_width // 2 + 100, board_height // 2 + 50))
    quit_button = pygame.Rect(quit_rect.left - 20, quit_rect.top - 10, quit_rect.width + 40, quit_rect.height + 20)

    pygame.draw.rect(screen, (0, 128, 0), replay_button)#draw rectangular shape
    pygame.draw.rect(screen, (128, 0, 0), quit_button)
    screen.blit(replay_text, replay_rect)
    screen.blit(quit_text, quit_rect) #update screen to show the button

    return replay_button, quit_button

def main():
    #initiate board engine
    model = joblib.load("chess_rf_model.joblib")  # load the pre trained model
    label_encoder = joblib.load("chess_label_encoder.joblib")  # load the lable encoder
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Skill Level": 5})

    initial_square = None
    dragging_piece = None
    draging_position = None
    game_over = False
    suggested = ""
    replay_button, quit_button = None, None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                engine.quit()
                pygame.quit()
                return
            #userw press mouse and it is user's turn
            elif not game_over and event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:
                x, y = event.pos
                square = pixel_to_square(x, y)
                piece = board.piece_at(square)
                if piece and piece.color == chess.WHITE:#if valid selections
                    initial_square = square
                    dragging_piece = piece
                    draging_position = event.pos
            elif not game_over and event.type == pygame.MOUSEMOTION and dragging_piece:# drag mouse
                draging_position = event.pos
            elif not game_over and event.type == pygame.MOUSEBUTTONUP and dragging_piece:#release mouse to complete move
                x, y = event.pos
                target_square = pixel_to_square(x, y)
                move = chess.Move(initial_square, target_square)
                if move in board.legal_moves:
                    board.push(move)
                    if not board.is_game_over():# stockfish's move
                        result = engine.play(board, chess.engine.Limit(time=0.01))
                        board.push(result.move)
                        suggested = str(suggest_move(board, model, label_encoder))

                initial_square = None
                dragging_piece = None
                draging_position = None
            #game over reset
            elif game_over and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if replay_button and replay_button.collidepoint(x, y):# if press replay
                    board = chess.Board()
                    game_over = False
                    initial_square = None
                    dragging_piece = None
                    draging_position = None
                elif quit_button and quit_button.collidepoint(x, y):#quit
                    engine.quit()
                    pygame.quit()
                    return
        #update board after each event
        print_board(screen, board, initial_square, dragging_piece, draging_position)
        comment_area = pygame.Rect(board_width, 0, comment_width, board_height)
        pygame.draw.rect(screen, (255, 255, 255), comment_area)
        font = pygame.font.Font(None, 30)
        text = font.render((suggested), True, (0, 0, 0))
        text_rect = text.get_rect(center=(board_width + comment_width / 2, board_height / 2))
        screen.blit(text, text_rect)
        # printw out messages if game over
        if board.is_game_over():
            game_over = True
            font = pygame.font.Font(None, 30)
            if board.is_checkmate():
                message = "White Wins" if board.turn == chess.BLACK else "Black Wins"
            elif board.is_stalemate():
                message = "Draw (Stalemate)"
            elif board.is_insufficient_material():
                message = "Draw (Insufficient Material)"
            else:
                message = "Draw (3 Repeated Moves)"
            #print out the messaages under game end
            text = font.render(message, True, (255, 0, 0))
            text_rect = text.get_rect(center=(board_width // 2, int(board_height // 2.3)))
            pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(board_width,20))
            screen.blit(text, text_rect)
            replay_button, quit_button = draw_buttons(screen)
        pygame.display.flip() #refresh board
    engine.quit()
    pygame.quit()

if __name__ == "__main__":
    main()