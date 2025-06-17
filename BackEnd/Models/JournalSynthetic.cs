using System;

namespace IndirectTax.Models
{
    public class JournalSynthetic
    {
        public string? Region { get; set; }
        public string? City { get; set; }
        public string? County { get; set; }
        public string? Entity { get; set; }
        public string? District { get; set; }
        public decimal Gross { get; set; }
        public decimal Taxable { get; set; }
        public string? InputSource { get; set; }
        public string? PlaceDetermination { get; set; }
        public int EntryId { get; set; }
        public string? Id { get; set; }
        public string? LogId { get; set; }
        public decimal TaxRate { get; set; }
        public int Year { get; set; }
        public int Month { get; set; }
    }
}
