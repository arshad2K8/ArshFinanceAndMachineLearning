import argparse

# Create the parser
parser = argparse.ArgumentParser(
    description="New home finances"
)

# Add arguments
parser.add_argument(
    "-v", "--home",
    type=int,
    required=True,
    help="Home Value"
)

parser.add_argument(
    "-d", "--deposit",
    type=int,
    required=True,
    help="Amount of Deposit "
)

parser.add_argument(
    "-o", "--over",
    type=float,
    required=True,
    help="Over Deposit "
)

parser.add_argument(
    "-i", "--rate",
    type=float,
    default=18,
    help="Mortgage rate"
)

parser.add_argument(
    "-y", "--years",
    type=int,
    default=18,
    help="Mortgage length in years"
)


def calculate_monthly_payment(principal, annual_interest_rate, years):
    """
    Calculate the monthly mortgage payment using the standard amortization formula.
    """
    monthly_rate = annual_interest_rate / 100 / 12
    num_payments = years * 12

    if monthly_rate == 0:  # handle zero-interest loans
        return principal / num_payments

    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return monthly_payment


def calculate_lbtt(price):
    """
    Calculate LBTT (Land and Buildings Transaction Tax) for residential properties in Scotland.
    Rates as of 2024:
      - Up to £145,000: 0%
      - £145,001 to £250,000: 2%
      - £250,001 to £325,000: 5%
      - £325,001 to £750,000: 10%
      - Over £750,000: 12%
    """
    bands = [
        (145000, 0.00),
        (250000, 0.02),
        (325000, 0.05),
        (750000, 0.10),
        (float('inf'), 0.12)
    ]
    
    lbtt = 0
    prev_limit = 0
    for limit, rate in bands:
        if price > limit:
            lbtt += (limit - prev_limit) * rate
            prev_limit = limit
        else:
            lbtt += (price - prev_limit) * rate
            break
    return lbtt

# Parse the arguments
args = parser.parse_args()
homePrice = args.home
deposit = args.deposit
over = args.over

annual_interest_rate = args.rate or 4.5
years = args.years or 25

print(f"Home Report Price is, {homePrice}!")

deposit_val = homePrice * (deposit / 100)
over_home_report = homePrice * (over / 100)
offer_price = homePrice + over_home_report
lbtt = calculate_lbtt(offer_price)

print("\n---Current Flat Summary ---")
curr_flat_price = 195000
my_flat_over_home_report = curr_flat_price * (5 / 100)
balance = 150000
equity = curr_flat_price  - balance
print(f"Current Flat HR:  £{curr_flat_price:,.2f}")
print(f"balance:          £{balance:,.2f}")
print(f"equity:           £{equity:,.2f}")


print("\n--- Calculation Summary ---")
print(f"Home Report Price:  £{homePrice:,.2f}")
print(f"Offer Price:        £{offer_price:,.2f}")
print(f"LBTT to pay:        £{lbtt:,.2f}")
print(f"{deposit}% Deposit:        £{deposit_val:,.2f}")
print(f"{over}% Over HR:       £{over_home_report:,.2f}")
print(f"Lawyer Fees:        £{1500:,.2f}")
estate_agent_fees = (curr_flat_price + my_flat_over_home_report) * (1 / 100)
print(f"{1}% Estate Agent fees:       £{estate_agent_fees:,.2f}")

total_cash_neeeded_now = lbtt + over_home_report+ deposit_val+ 1500 + estate_agent_fees
short_of = total_cash_neeeded_now - equity
print(f"\nTotal Cash ( LBTT + over_home_report + Deposit + Lawyer + estate agent): £{total_cash_neeeded_now:,.2f}")
print(f"Total Cash Needed Now:   £{total_cash_neeeded_now:,.2f}")
print(f"Equity Currently:        £{equity:,.2f}")
print(f"SHORT OF:                £{short_of:,.2f}")


principal = homePrice - deposit_val
monthly_payment = calculate_monthly_payment(principal, annual_interest_rate, years)
total_payment = monthly_payment * years * 12
total_interest = total_payment - principal
print("\n--- Mortgage Summary ---")
print(f"Home Report Price:    £{homePrice:,.2f}")
print(f"deposit_val:          £{deposit_val:,.2f}")
print(f"Loan Amount:          £{principal:,.2f}")
print(f"Interest Rate:        {annual_interest_rate:.2f}%")
print(f"Term:                 {years} years")
print(f"Monthly Payment:      £{monthly_payment:,.2f}")
print(f"Total Payment:        £{total_payment:,.2f}")
print(f"Total Interest Paid:  £{total_interest:,.2f}\n")