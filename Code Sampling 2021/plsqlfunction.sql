CREATE OR REPLACE FUNCTION get_stock_price(v_stockTicker IN stock.stockTicker%TYPE) 
RETURN NUMBER IS   
v_currentPrice  stock.sCurrentPrice%TYPE :=  0;
BEGIN   
	SELECT sCurrentPrice INTO v_currentPrice   
	FROM Stock
	WHERE stockTicker =v_stockTicker; 
	RETURN (v_currentPrice);
END get_stock_price;

VARIABLE get_stockPrice number;
exec :get_stockPrice := get_stock_price('GME');
PRINT get_stockPrice;


Show ERRORS FUNCTION num_of_clients;