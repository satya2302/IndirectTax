import { Component, Input, OnChanges } from '@angular/core';

@Component({
  selector: 'app-usage-chart',
  standalone: true,
  template: `
    <div style="width:100%;max-width:1400px;margin:auto;">
      <canvas id="usageChart" height="320" width="1300"></canvas>
    </div>
  `
})
export class UsageChartComponent implements OnChanges {
  @Input() usageList: any[] = [];

  ngOnChanges() {
    this.renderChart();
  }

  renderChart() {
    if (!(window as any).Chart) return;
    const ctx = (document.getElementById('usageChart') as HTMLCanvasElement)?.getContext('2d');
    if (!ctx) return;
    if ((window as any).usageChartInstance) {
      (window as any).usageChartInstance.destroy();
    }
    const labels = this.usageList.map(u => `${u.year}-${u.month}`);
    const transactions = this.usageList.map(u => u.transactions);
    const taxReturns = this.usageList.map(u => u.taxReturns);
    const eFilings = this.usageList.map(u => u.eFilings);
    (window as any).usageChartInstance = new (window as any).Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Transactions', data: transactions, backgroundColor: '#42a5f5' },
          { label: 'Tax Returns', data: taxReturns, backgroundColor: '#66bb6a' },
          { label: 'EFilings', data: eFilings, backgroundColor: '#ffa726' }
        ]
      },
      options: {
        responsive: false,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: { x: { stacked: true }, y: { stacked: true } },
        animation: false
      }
    });
  }
}
