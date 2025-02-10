
def get_filters():
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT Model, Name, Vin, Make, Year, Miles, Exterior_Color, Interior_Color, Option, Fuel_Type FROM cars", conn)
    conn.close()
    return df.to_json(orient='records')

def get_filter_values(filter_type):
    valid_filters = ["Model", "Name", "Make", "Year", "Miles", "Exterior_Color", "Interior_Color", "Option", "Fuel_Type"]
    
    if filter_type not in valid_filters:
        return jsonify({"error": "Invalid filter type"}), 400
    
    conn = sqlite3.connect('data.db')
    query = f"SELECT DISTINCT {filter_type} FROM cars"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return jsonify(df[filter_type].dropna().tolist())

def get_database():
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT * FROM cars", conn)
    conn.close()
    return df.to_json(orient='records')

def filter_database():
    model = request.args.get('model')
    name = request.args.get('name')
    vin = request.args.get('vin')
    make = request.args.get('make')
    year = request.args.get('year')
    miles = request.args.get('miles')
    exterior_color = request.args.get('exterior_color')
    interior_color = request.args.get('interior_color')
    option = request.args.get('option')
    fuel_type = request.args.get('fuel_type')

    query = "SELECT * FROM cars WHERE 1=1"
    params = {}

    if model:
        query += " AND Model = :model"
        params['model'] = model
    if name:
        query += " AND Name = :name"
        params['name'] = name
    if vin:
        query += " AND Vin = :vin"
        params['vin'] = vin
    if make:
        query += " AND Make = :make"
        params['make'] = make
    if year:
        query += " AND Year = :year"
        params['year'] = year
    if miles:
        query += " AND Miles = :miles"
        params['miles'] = miles
    if exterior_color:
        query += " AND Exterior_Color = :exterior_color"
        params['exterior_color'] = exterior_color
    if interior_color:
        query += " AND Interior_Color = :interior_color"
        params['interior_color'] = interior_color
    if option:
        query += " AND Option = :option"
        params['option'] = option
    if fuel_type:
        query += " AND Fuel_Type = :fuel_type"
        params['fuel_type'] = fuel_type

    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df.to_json(orient='records')