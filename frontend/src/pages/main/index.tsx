import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/shared/ui/chart';
import { type FC } from 'react';
import { Bar, BarChart, CartesianGrid } from 'recharts';

const chartData = [
  { month: 'January', desktop: 186, mobile: 80 },
  { month: 'February', desktop: 305, mobile: 200 },
  { month: 'March', desktop: 237, mobile: 120 },
  { month: 'April', desktop: 73, mobile: 190 },
  { month: 'May', desktop: 209, mobile: 130 },
  { month: 'June', desktop: 214, mobile: 140 },
];

const chartConfig = {
  desktop: {
    label: 'Desktop',
    color: '#2563eb',
  },
  mobile: {
    label: 'Mobile',
    color: '#60a5fa',
  },
} satisfies ChartConfig;

const MainPage: FC = () => (
  <ChartContainer config={chartConfig} className='min-h-[200px] w-auto h-full'>
    <BarChart accessibilityLayer data={chartData}>
      <CartesianGrid />
      <ChartTooltip content={<ChartTooltipContent />} />
      <ChartLegend content={<ChartLegendContent />} />
      <Bar dataKey='desktop' fill='var(--color-desktop)' radius={4} />
      <Bar dataKey='mobile' fill='var(--color-mobile)' radius={4} />
    </BarChart>
  </ChartContainer>
);

export default MainPage;
