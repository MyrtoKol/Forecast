import sqlite3

def create_forecast_table(database_path):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(database_path)

    # Create a cursor object to execute SQL queries
    c = conn.cursor()

    # Create a table named 'forecast' with two columns: 'timestamp' and 'value'
    c.execute('''CREATE TABLE IF NOT EXISTS forecast (
                    timestamp TEXT,
                    value REAL
                )''')

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()