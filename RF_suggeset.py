import chess
import chess.engine
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import kagglehub
import pandas as pd
import os
import joblib

path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
print("Path to dataset files:", path)
engine_path = "stockfish.exe"


# Helper function to calculate piece mobility (legal moves for each piece)
def piece_mobility(board):
    mobility_feats = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            legal_moves = [move for move in board.legal_moves if move.from_square == square]
            mobility_feats.append(len(legal_moves))  # Number of legal moves
        else:
            mobility_feats.append(0)
    return mobility_feats

def piece_value(piece):
    if piece is None:
        return 0
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type == chess.KNIGHT or piece.piece_type == chess.BISHOP:
        return 3
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9
    elif piece.piece_type == chess.KING:
        return 1000
    return 0

# Helper function to calculate total material value for each side
def material_balance(board):
    white_value = 0
    black_value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_val = piece_value(piece)
            if piece.color == chess.WHITE:
                white_value += piece_val
            else:
                black_value += piece_val
    return white_value, black_value

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
    features.append(white_value)
    features.append(black_value)
    features.append(board.fullmove_number)
    features.append(board.halfmove_clock)
    features.append(int(board.is_check()))
    features.append(int(board.is_checkmate()))
    features.append(int(board.is_stalemate()))

    return np.array(features)

def center_control_features(board):
    # Define central squares
    core_center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    extended_center_squares = [
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6
    ]
    features = []
    white_occupies_core = sum(
        1 for sq in core_center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
    )
    black_occupies_core = sum(
        1 for sq in core_center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
    )
    white_occupies_extended = sum(
        1 for sq in extended_center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
    )
    black_occupies_extended = sum(
        1 for sq in extended_center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
    )

    features.append(white_occupies_core)
    features.append(black_occupies_core)
    features.append(white_occupies_extended)
    features.append(black_occupies_extended)
    white_controls_core = sum(
        1 for sq in core_center_squares if board.is_attacked_by(chess.WHITE, sq)
    )
    black_controls_core = sum(
        1 for sq in core_center_squares if board.is_attacked_by(chess.BLACK, sq)
    )

    white_controls_extended = sum(
        1 for sq in extended_center_squares if board.is_attacked_by(chess.WHITE, sq)
    )
    black_controls_extended = sum(
        1 for sq in extended_center_squares if board.is_attacked_by(chess.BLACK, sq)
    )

    features.append(white_controls_core)
    features.append(black_controls_core)
    features.append(white_controls_extended)
    features.append(black_controls_extended)

    return features

def defense_features(board):
    defense_feats = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)  # Pieces attacking this square
            defenders = board.attackers(piece.color, square)     # Pieces defending this square

            if len(attackers) > 0:
                is_defended = len(defenders) > 0
            else:

                is_defended = True

            defense_feats.append(1 if is_defended else 0)
        else:
            defense_feats.append(0)

    return defense_feats

def capture_features(board):
    capture_feats = np.zeros(64, dtype=int)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attackers = board.attackers(not piece.color, square)
            defenders = board.attackers(piece.color, square)
            # Number of attackers minus defenders
            net_threat = len(attackers) - len(defenders)

            if net_threat > 0:
                capture_feats[square] = 1
            elif net_threat < 0:
                capture_feats[square] = -1

    return capture_feats

def load_training_data(dataset_path, engine, sample_size):
    Trainfile = None
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                Trainfile = os.path.join(root, file)
                break
    datafile = pd.read_csv(Trainfile)
    datafile = datafile.sample(n=sample_size, random_state=42)

    X, y = [], []
    for idx, row in datafile.iterrows():
        fen = row["FEN"]
        board = chess.Board(fen)

        features = board_features(board)

        result = engine.play(board, chess.engine.Limit(depth=4))
        move = result.move
        if move is None:
            continue

        X.append(features)
        y.append(move.uci())

    return np.array(X), y

def trainmodel():

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Skill Level": 15})
    #load
    X, y = load_training_data(path, engine, 5000)
    label_encoder = LabelEncoder()
    #train
    y_encoded = label_encoder.fit_transform(y)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y_encoded)
    #save
    joblib.dump(model, "chess_rf_model.joblib")
    joblib.dump(label_encoder, "chess_label_encoder.joblib")
    engine.quit()
if __name__ == "__main__":
    trainmodel()