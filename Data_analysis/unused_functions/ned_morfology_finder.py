# from astroquery.ipac.ned import Ned
# from astropy.coordinates import SkyCoord
# import astropy.units as u

# coord = SkyCoord(ra=49.9507*u.deg, dec=41.5117*u.deg, frame='icrs')

# result = Ned.query_region(coord, radius=5*u.arcsec)

# obj=result[result['Separation'].argmin()]

# obj_name=obj['Object Name']

# classif = Ned.query_object_async(obj_name,get_query_payload=True)
# print(classif)
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
import astropy.units as u

coord = SkyCoord(ra=49.9507*u.deg, dec=41.5117*u.deg, frame='icrs')

# Busca objetos na região
result = Ned.query_region(coord, radius=5*u.arcsec)

# Pega o objeto mais próximo
obj = result[result['Separation'].argmin()]
obj_name = obj['Object Name']

# Método 1: Query object simples
print("=== Método 1: Query Object ===")
try:
    obj_data = Ned.query_object(obj_name)
    print(obj_data)
except Exception as e:
    print(f"Erro: {e}")

# Método 2: Search by identifier
print("\n=== Método 2: Search by Identifier ===")
try:
    search_results = Ned.search_by_object(obj_name)
    print(search_results)
except Exception as e:
    print(f"Erro: {e}")

# Método 3: Get all tables e procurar por classificação
print("\n=== Método 3: Procurando em todas as tabelas ===")
try:
    # Lista todas as tabelas disponíveis
    table_names = Ned.get_table_names(obj_name)
    print(f"Tabelas disponíveis: {table_names}")
    
    # Procura em cada tabela por informações de classificação
    for table_name in table_names:
        try:
            table_data = Ned.get_table(obj_name, table=table_name)
            # Procura colunas relacionadas a classificação
            classification_cols = [col for col in table_data.colnames 
                                 if 'type' in col.lower() or 
                                    'class' in col.lower() or
                                    'morph' in col.lower()]
            if classification_cols:
                print(f"\nTabela: {table_name}")
                for col in classification_cols:
                    print(f"  {col}: {table_data[col][0]}")
        except:
            continue
except Exception as e:
    print(f"Erro: {e}")