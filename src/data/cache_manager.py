import os
import sqlite3
from typing import List, Optional
import pandas as pd


class SQLitePriceCache:
    def __init__(self, db_filename: str = "market_cache.db"):
        # create data directory safely in project workspace
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, db_filename)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        return conn

    def _init_database(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS asset_prices (
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            adj_close REAL NOT NULL,
            PRIMARY KEY (ticker, trade_date)
        );
        """
        cursor.execute(create_table_query)

        # index for fast lookup by date range
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_date ON asset_prices (ticker, trade_date);"
        )

        conn.commit()
        conn.close()

    def get_cached_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        conn = self._get_connection()
        placeholders = ",".join(["?"] * len(tickers))
        
        query = f"""
        SELECT trade_date, ticker, adj_close
        FROM asset_prices
        WHERE ticker IN ({placeholders})
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date ASC;
        """

        params = list(tickers) + [start_date, end_date]
        df_records = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df_records.empty:
            return pd.DataFrame()

        # pivot table to have dates as index and tickers as columns
        pivot_df = df_records.pivot(index="trade_date", columns="ticker", values="adj_close")
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df.sort_index()

        return pivot_df

    def save_prices(self, price_df: pd.DataFrame) -> None:
        if price_df.empty:
            return

        records_to_insert = []
        for dt_idx, row in price_df.iterrows():
            date_str = pd.to_datetime(dt_idx).strftime("%Y-%m-%d")
            for ticker, price_val in row.items():
                if pd.notna(price_val):
                    records_to_insert.append((str(ticker), date_str, float(price_val)))

        if not records_to_insert:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        insert_query = """
        INSERT OR REPLACE INTO asset_prices (ticker, trade_date, adj_close)
        VALUES (?, ?, ?);
        """
        cursor.executemany(insert_query, records_to_insert)
        conn.commit()
        conn.close()