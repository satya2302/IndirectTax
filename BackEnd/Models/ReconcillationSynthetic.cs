using System;

namespace IndirectTax.Models
{
    public class ReconcillationSynthetic
    {
        public string Region { get; set; }
        public string City { get; set; }
        public string County { get; set; }
        public string Entity { get; set; }
        public decimal Gross { get; set; }
        public decimal Taxable { get; set; }
        public decimal UnreportedTax { get; set; }
        public decimal TaxRate { get; set; }
        public int Year { get; set; }
        public int Month { get; set; }
    }
}