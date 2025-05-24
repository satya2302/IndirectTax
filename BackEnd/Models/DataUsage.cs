using System;

namespace IndirectTax.Models
{
    public class DataUsage
    { 

         public decimal year { get; set; }
         public string Month { get; set; }
        public decimal Transactions { get; set; }
        public decimal TaxReturns { get; set; }
        public decimal EFilings { get; set; }

        public decimal Users_Transactions { get; set; }
        public decimal Users_TaxReturns { get; set; }
        public decimal Users_EFilings { get; set; }
    }
        // Add other properties as per your Datausage table
}
