from llama_index.core.tools.tool_spec.base import BaseToolSpec
from rich.console import Console
from typing import List
import yfinance as yf
import pandas as pd


class StockAnalystsToolSpec(BaseToolSpec):
    """Stock analysis tool spec. using Yahoo Finance"""

    spec_functions = []

    def __init__(self) -> None:
        """initialize the tool"""
        self.console = Console()

    # helper functions
    def __download_data(self, ticker: str) -> List:
        """
        downloads stock data from Yahoo Finance and returns
        ticker data, financials, balance sheet, cash flow.
        The financials, balance sheet & cash flows are sorted by
        date in ascending order, so that most recent year is the last.

        Args:
            ticker(str): the stock ticker to be given to yfinance
        """
        ticker_data = yf.Ticker(ticker)
        financials = ticker_data.financials.transpose()
        balance_sheet = ticker_data.balance_sheet.transpose()
        cash_flow = ticker_data.cashflow.transpose()

        # NOTE: financials, balance_sheet and cash_flow are by default
        # reverse sorted by date-time index (i.e. have the
        # most recent year on top). This screws up all calculations
        # We'll fix that by sorting dataframes in ascending data order
        # (i.e. latest year is the last in the dataframe)
        financials = financials.sort_index(ascending=True)
        balance_sheet = balance_sheet.sort_index(ascending=True)
        cash_flow = cash_flow.sort_index(ascending=True)

        return ticker_data, financials, balance_sheet, cash_flow

    def financial_ratios(self, ticker: str) -> pd.DataFrame:
        """
        calculates all the typical financial ratios that a financial analyst
        will use to analyze a company's stock, such as liquidity ratios,
        profitability ratios, efficiency ratios, valuation ratioe

        Args:
            ticker(str): the stock ticker to be given to yfinance

        Returns:
            pandas dataframe of ratios with index = ratio name & value
            is the calculated value for the ratio.
            All calculations are for the entire data that is downloaded
            from Yahoo Finance
        """
        ticker_data, financials, balance_sheet, cash_flow = self.__download_data(ticker)

        ratios = {}

        info = ticker_data.info

        try:
            # Balance Sheet
            current_assets = balance_sheet["Current Assets"].iloc[-1]
            current_liabilities = balance_sheet["Current Liabilities"].iloc[-1]
            total_assets = balance_sheet["Total Assets"].iloc[-1]
            shareholder_equity = balance_sheet["Stockholders Equity"].iloc[-1]
            total_debt = balance_sheet["Total Debt"].iloc[-1]

            # Financials
            revenue = financials["Total Revenue"].iloc[0]
            operating_income = financials["Operating Income"].iloc[0]
            net_income = financials["Net Income"].iloc[0]
            cost_of_goods_sold = financials["Cost Of Revenue"].iloc[0]
            ebit = financials["EBIT"].iloc[0]

            # Inventory Data
            # NOTE: inventory may or may not get reported. For example,
            # Tata Motors reports it, Persisteny Systems does not
            inventory_fields_exist = "Inventory" in ticker_data.balance_sheet.index
            if inventory_fields_exist:
                inventory_current = balance_sheet["Inventory"].iloc[-1]
                inventory_previous = balance_sheet["Inventory"].iloc[-2]
                average_inventory = (inventory_current + inventory_previous) / 2

            # Interest Expense
            interest_expense = financials["Interest Expense"].iloc[0]

            # Ratios -------------------------------

            ratios["Revenue Growth"] = financials["Total Revenue"].pct_change()
            ratios["Earnings-per-share (EPS)"] = (
                financials["Net Income"] / balance_sheet["Common Stock"]
            )

            # Liquidity Ratios
            ratios["Current Ratio"] = current_assets / current_liabilities
            if inventory_fields_exist:
                ratios["Quick Ratio"] = (
                    current_assets - balance_sheet["Inventory"].iloc[0]
                ) / current_liabilities
            # Profitability Ratios
            ratios["Net Profit Margin"] = net_income / revenue
            ratios["Operating Margin"] = operating_income / revenue
            ratios["Return on Assets (RoA)"] = net_income / total_assets
            ratios["Return on Equity (RoE)"] = net_income / shareholder_equity
            # Leverage Ratios
            ratios["Debt-to-Equity (D/E)"] = total_debt / shareholder_equity
            ratios["Interest Coverage"] = ebit / interest_expense

            # Efficiency ratios
            ratios["Asset Turnover"] = revenue / total_assets
            if inventory_fields_exist:
                ratios["Inventory Turnover"] = cost_of_goods_sold / average_inventory

            # Valuation Ratios
            market_cap = info["marketCap"]
            ratios["Price-to-Earnings (P/E)"] = info["trailingPE"]
            ratios["Price-to-Sales (P/S)"] = market_cap / revenue
            ratios["Price-to-Book (P/B)"] = market_cap / shareholder_equity

            return pd.DataFrame(ratios)
        except KeyError as e:
            self.console.print(f"[red]KeyError: missing data for {e}[/red]")
            raise e


def display_ratios(sat: StockAnalystsToolSpec, ticker: str) -> None:
    console.print(f"[green]Financial ratios for {ticker}[/green]")
    df = sat.financial_ratios(ticker)
    console.print(df)


if __name__ == "__main__":
    sat = StockAnalystsToolSpec()
    console = Console()
    # show me ratios for Microsoft
    tickers = ["MSFT", "AAPL", "AMZN", "RELIANCE.NS", "PIDILITIND.NS", "COLPAL.NS"]
    for ticker in tickers[:1]:
        display_ratios(sat, ticker)
